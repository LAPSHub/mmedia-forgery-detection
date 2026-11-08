import os
import glob
import math
import time
import warnings
import numpy as np
import pandas as pd             
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast 
import timm
import matplotlib.pyplot as plt
import seaborn as sns           
from tqdm.notebook import tqdm            
from PIL import Image

# --- MÉTRICAS (SCIKIT-LEARN) ---
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,                  
    auc                         
)

# Filtra avisos desnecessários
warnings.filterwarnings("ignore")

# Configuração para/kaggle/input/dataseht-40frames/cache_frames_F3Net_FaceShifter_40frames salvar gráficos sem abrir janela
plt.switch_backend('Agg') 


# Caminho base onde seus tensores .npy estão (Input - Read Only)
BASE_KAGGLE = '/kaggle/input/datasets/marinaldolima/dataset-test-2/dfdc-test'

CACHE_TRAIN = os.path.join(BASE_KAGGLE, 'train')
CACHE_VAL   = os.path.join(BASE_KAGGLE, 'val')
CACHE_TEST  = os.path.join(BASE_KAGGLE, 'test')

SAVE_DIR = '/kaggle/working/weights'
os.makedirs(SAVE_DIR, exist_ok=True)


#SAVE_DIR_2 = '/kaggle/input/models/marinaldolima/test-f3-net/pytorch/default/1'
#os.makedirs(SAVE_DIR_2, exist_ok=True)


# Verificação rápida
for p in [CACHE_TRAIN, CACHE_VAL, CACHE_TEST]:
    if not os.path.exists(p):
        print(f" AVISO: Caminho não encontrado: {p} (Verifique o nome do dataset na direita)")
    else:
        print(f" Pasta encontrada: {p}")

# ==============================================================================

WEIGHT_DECAY = 0.5e-4           
DROPOUT_RATE = 0.3     
NUM_EPOCHS = 7          
BATCH_SIZE_VIDEOS = 4      
FACES_PER_VIDEO = 20
MOMENTUM = 0.9
FACTOR = 0.5
INPUT_SIZE = (299, 299)
LEARNING_RATE = 0.2e-3
ACC_STEPS = 8
PATIENCE = 10


# Detecção de GPU (Crucial no Kaggle)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n--- Configuração de Hardware ---")
print(f" Dispositivo: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   Placa de Vídeo: {torch.cuda.get_device_name(0)}")
else:
    print(" ATENÇÃO: Você está na CPU! Ative a GPU nas configurações do Kaggle.")

print(f" Lendo dados de: {BASE_KAGGLE}")
print(f" Salvando resultados em: {SAVE_DIR}")
print("-" * 30)

# ==============================================================================

def sigma(x):
    return torch.tanh(x / 2.0)

# Classe DCT para PyTorch (Transformada Discreta de Cosseno)
class DCT(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.register_buffer('dct_matrix', self._get_dct_matrix(N))

    def _get_dct_matrix(self, N):
        dct_m = torch.zeros((N, N))
        for k in range(N):
            for n in range(N):
                if k == 0: dct_m[k, n] = 1 / math.sqrt(N)
                else: dct_m[k, n] = math.sqrt(2 / N) * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
        return dct_m

    def forward(self, x):
        x_dct = torch.matmul(self.dct_matrix, x)
        x_dct = torch.matmul(x_dct, self.dct_matrix.t())
        return x_dct

    def inverse(self, x):
        x_idct = torch.matmul(self.dct_matrix.t(), x)
        x_idct = torch.matmul(x_idct, self.dct_matrix)
        return x_idct

# --- FAD LAYER (Frequência) ---
class FAD_Layer(nn.Module):
    def __init__(self, size=299): 
        super().__init__()
        self.dct = DCT(size)
        u = torch.arange(size).view(-1, 1).float()
        v = torch.arange(size).view(1, -1).float()
        freq_mag = torch.sqrt(u**2 + v**2)
        
        limit_low = size // 8; limit_mid = size // 4
        masks = torch.zeros(3, size, size)
        masks[0] = (freq_mag < limit_low).float()
        masks[1] = ((freq_mag >= limit_low) & (freq_mag < limit_mid)).float()
        masks[2] = (freq_mag >= limit_mid).float()
        
        self.register_buffer('f_base', masks)
        self.f_w = nn.Parameter(torch.zeros(1, 9, 1, 1)) 

    def forward(self, x):
        x_dct = self.dct(x) 
        outputs = []
        for i in range(3):
            mask = self.f_base[i].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(x.device)
            w = self.f_w[:, i*3 : (i+1)*3, :, :]
            filter_factor = mask + sigma(w) * mask
            dct_filtered = x_dct * filter_factor
            outputs.append(self.dct.inverse(dct_filtered))
        return torch.cat(outputs, dim=1)

# --- LFS LAYER (Textura) ---
class LFS_Layer(nn.Module):
    def __init__(self, window_size=10, stride=2, M=6):
        super().__init__()
        self.window_size = window_size; self.stride = stride; self.M = M
        self.dct = DCT(window_size)
        
        u = torch.arange(window_size).view(-1, 1).float()
        v = torch.arange(window_size).view(1, -1).float()
        idx_sum = u + v
        limits = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 18)]
        
        masks = torch.zeros(M, window_size, window_size)
        for i in range(M):
            low, high = limits[i]
            masks[i] = ((idx_sum >= low) & (idx_sum <= high)).float()
            
        self.register_buffer('lfs_masks', masks)
        self.h_w = nn.Parameter(torch.zeros(1, 3*M, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.window_size, stride=self.stride, padding=0) 
        L_patches = patches.size(2)
        patches = patches.view(B, C, self.window_size, self.window_size, L_patches).permute(0, 4, 1, 2, 3).contiguous() 
        patches = patches.view(B * L_patches, C, self.window_size, self.window_size)
        
        dct_patches = self.dct(patches)
        features_list = []
        for i in range(self.M):
            mask = self.lfs_masks[i].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(x.device)
            l1_norm = torch.sum(torch.abs(dct_patches * mask), dim=(2, 3)) 
            features_list.append(torch.log10(l1_norm + 1e-10))
            
        stats_total = torch.cat(features_list, dim=1)
        w_expanded = 1.0 + sigma(self.h_w).view(1, 18)
        stats_total = stats_total + torch.log10(w_expanded + 1e-10)
        
        H_out = (H - self.window_size) // self.stride + 1
        W_out = (W - self.window_size) // self.stride + 1
        return stats_total.view(B, H_out, W_out, 3 * self.M).permute(0, 3, 1, 2)

class MixBlock(nn.Module):
    def __init__(self, C_in):
        super().__init__()
        self.C_mid = max(C_in // 4, 32) 
        self.scale = self.C_mid ** -0.5
        self.conv_f_q = nn.Conv2d(C_in, self.C_mid, 1); self.conv_f_k = nn.Conv2d(C_in, self.C_mid, 1); self.conv_f_v = nn.Conv2d(C_in, C_in, 1)
        self.conv_l_q = nn.Conv2d(C_in, self.C_mid, 1); self.conv_l_k = nn.Conv2d(C_in, self.C_mid, 1); self.conv_l_v = nn.Conv2d(C_in, C_in, 1)

    def forward(self, F_in, L_in):
        B, C, H, W = F_in.shape; HW = H * W
        FQ = self.conv_f_q(F_in).view(B, self.C_mid, HW).transpose(1,2)
        FK = self.conv_f_k(F_in).view(B, self.C_mid, HW)
        FV = self.conv_f_v(F_in).view(B, C, HW)
        LQ = self.conv_l_q(L_in).view(B, self.C_mid, HW).transpose(1,2)
        LK = self.conv_l_k(L_in).view(B, self.C_mid, HW)
        LV = self.conv_l_v(L_in).view(B, C, HW)
        A_LF = F.softmax(torch.matmul(LQ, FK) * self.scale, dim=-1)
        A_FL = F.softmax(torch.matmul(FQ, LK) * self.scale, dim=-1)
        L_aug = torch.matmul(A_LF, FV.transpose(1,2)).transpose(1,2).view(B,C,H,W)
        F_aug = torch.matmul(A_FL, LV.transpose(1,2)).transpose(1,2).view(B,C,H,W)
        return F_in + F_aug, L_in + L_aug

def get_xception_backbone(in_channels):
    model = timm.create_model('xception', pretrained=True, features_only=False)
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        new_conv1.weight[:, :3] = original_conv1.weight
        if in_channels > 3:
            avg_weight = original_conv1.weight.mean(dim=1, keepdim=True)
            new_conv1.weight[:, 3:] = avg_weight.repeat(1, in_channels - 3, 1, 1)
    model.conv1 = new_conv1
    return model



class F3Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fad = FAD_Layer()
        self.lfs = LFS_Layer()
        self.backbone_f = get_xception_backbone(9)
        self.backbone_l = get_xception_backbone(18)
        self.mix1 = MixBlock(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(512, 1)
            
        )

    def forward(self, x):
        f_in = self.fad(x)
        l_in = self.lfs(x)
        f = self.backbone_f.forward_features(f_in)
        l = self.backbone_l.forward_features(l_in)
        
        if f.shape[2:] != l.shape[2:]: 
            l = F.interpolate(l, size=f.shape[2:])
            
        f, l = self.mix1(f, l)
        
        
        feat = torch.cat([self.pool(f).flatten(1), self.pool(l).flatten(1)], 1)
        
        return self.classifier(feat)

# ==============================================================================
# 4. DATASET KAGGLE (Leitura Direta de Tensores)
# ==============================================================================
def auditoria_dataset(loader, model, device, desc="Audit"):
    model.eval()
    correct_v = 0
    total_v = 0
    correct_f = 0
    total_f = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            if batch is None: continue
            imgs, lbls, _ = batch
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            with autocast():
                out = model(imgs)
            
            # Aplica Sigmoid e Round pois a F3Net retorna Logits
            probs = torch.sigmoid(out)
            preds = torch.round(probs)
            
            # --- Frame Metric ---
            correct_f += (preds == lbls).sum().item()
            total_f += lbls.numel()

            # --- Video Metric (Seguro pois o loader tem shuffle=False) ---
            for k in range(0, len(preds), FACES_PER_VIDEO):
                v_preds = preds[k : k + FACES_PER_VIDEO]
                v_lbl = lbls[k].item()
                # Regra: > 50% dos frames
                is_fake_pred = v_preds.sum().item() > (FACES_PER_VIDEO / 2)
                if (1 if is_fake_pred else 0) == v_lbl:
                    correct_v += 1
                total_v += 1
                
    acc_video = (correct_v / total_v) if total_v > 0 else 0
    acc_frame = (correct_f / total_f) if total_f > 0 else 0
    return acc_video, acc_frame

class VideoFaceDataset(Dataset):
    def __init__(self, cache_dir, frames_per_video=FACES_PER_VIDEO, mode='train'):
        self.cache_dir = cache_dir
        self.frames_per_video = frames_per_video
        self.mode = mode
        self.file_paths = []
        self.labels = []
        
        classes = {'real': 0, 'fake': 1}
        print(f"🔍 [{mode.upper()}] Carregando tensores de: {cache_dir}")
        
        for cls_name, label in classes.items():
            cls_folder = os.path.join(cache_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
            
            # Busca todos os .npy
            files = sorted(glob.glob(os.path.join(cls_folder, "*.npy")))
            self.file_paths.extend(files)
            self.labels.extend([label] * len(files))
            
        print(f"   -> Sucesso: {len(self.file_paths)} arquivos encontrados.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            npy_path = self.file_paths[idx]
            label_int = self.labels[idx]
            
            # 1. Carrega Tensor
            video_tensor = np.load(npy_path) # Shape: [Frames, H, W, 3] ou [Frames, H, W, C]
            total_frames = video_tensor.shape[0]

            # 2. Seleção de Frames
            if self.mode == 'train':
                replace = False if total_frames >= self.frames_per_video else True
                indices = np.random.choice(total_frames, self.frames_per_video, replace=replace)
                indices = np.sort(indices)
                chosen_frames = video_tensor[indices]
            else:
                indices = np.linspace(0, total_frames-1, self.frames_per_video, dtype=int)
                chosen_frames = video_tensor[indices]

            # 3. Converte para PyTorch [C, H, W]
            faces = torch.from_numpy(chosen_frames).permute(0, 3, 1, 2).float()
            
            # 4. Normaliza se precisar (Se max > 1.0, divide por 255)
            # Como seu pré-processamento já dividiu, isso aqui é só segurança
            if faces.max() > 1.0:
                faces = faces / 255.0

            # 5. Normalização ImageNet (Média e Desvio Padrão)
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            faces = (faces - mean) / std
            
            label_tensor = torch.full((self.frames_per_video,), label_int, dtype=torch.float32)

            return faces, label_tensor, npy_path

        except Exception as e:
            return None, None, None

def collate_fn_train(b):
    b = [x for x in b if x is not None and x[0] is not None]
    if not b: return None
    imgs = torch.cat([x[0] for x in b])
    lbls = torch.cat([x[1] for x in b]).unsqueeze(1)
    return imgs, lbls, None

def collate_fn_test(b):
    b = [x for x in b if x is not None and x[0] is not None]
    if not b: return None
    imgs = torch.cat([x[0] for x in b])
    lbls = torch.cat([x[1] for x in b]).unsqueeze(1)
    names = [x[2] for x in b]
    return imgs, lbls, names

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from torch.cuda.amp import GradScaler, autocast # Importante para economizar memória
# --- ADICIONE ESSAS VARIÁVEIS NO SEU BLOCO DE CONFIGURAÇÃO ---

def main_train():
    print(f"📂 Preparando Datasets...")
    # Lembre-se que o VideoFaceDataset deve converter o npy para tensor e garantir o shape (40, 3, 299, 299)
    train_ds = VideoFaceDataset(cache_dir=CACHE_TRAIN, frames_per_video=FACES_PER_VIDEO, mode='train')
    val_ds   = VideoFaceDataset(cache_dir=CACHE_VAL, frames_per_video=FACES_PER_VIDEO, mode='val')

    if len(train_ds) == 0:
        print("❌ Nenhum arquivo encontrado."); return
    
    # --- CONFIGURAÇÃO DE BATCH E ACUMULAÇÃO ---
    mini_batch = BATCH_SIZE_VIDEOS 
    accumulation_steps = ACC_STEPS # Variável configurável
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_VIDEOS, shuffle=True, 
        num_workers=2, pin_memory=True, collate_fn=collate_fn_train)

    train_audit_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_VIDEOS, shuffle=False, 
        num_workers=2, pin_memory=True, collate_fn=collate_fn_train)

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE_VIDEOS, shuffle=False, 
        num_workers=2, pin_memory=True, collate_fn=collate_fn_train)
    
    print(f"🚀 Iniciando F3Net (Batch Efetivo: {mini_batch * accumulation_steps}) no device: {DEVICE}...")
    model = F3Net().to(DEVICE)
    PATH_PESOS = '/kaggle/working/weights/latest.pth' 
    
    if os.path.exists(PATH_PESOS):
        # O map_location evita erro de GPU cruzada
        state_dict = torch.load(PATH_PESOS, map_location=DEVICE)
        
        # TRUQUE DE MESTRE: Remove o prefixo 'module.' se ele existir no arquivo salvo
        # Isso garante que carrega independente de ter sido salvo com 1 ou 2 GPUs antes
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict)
        print(f"✅ SUCESSO: Pesos carregados!")
    else:
        print("⚠️ AVISO: Começando do zero.")

    # --- CONFIGURAÇÃO de tudo descongelado
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Métricas de Parâmetros
    num_total = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_frozen = num_total - num_trainable

    print(f"Total de Parâmetros: {num_total:,}")
    print(f"Treináveis: {num_trainable:,}")
    print(f"Congelados: {num_frozen:,}")
    print("-" * 40)
    print("DETALHAMENTO POR BLOCO:")

    # Loop para pegar cada parte declarada no self (FAD, LFS, Xception, Classifier, etc)
    for name, child in model.named_children():
        qtd_params = sum(p.numel() for p in child.parameters())
        print(f"  🔹 {name}: {qtd_params:,} parâmetros")
    
    print("-" * 40)
    # Otimizador e Scaler para FP16
    opt = optim.SGD(trainable_params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
   
    scaler = GradScaler() 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=FACTOR, patience=2)
    crit = nn.BCEWithLogitsLoss()
    
    history = {
        't_loss': [], 
        'v_loss': [],
        't_acc_video': [], 
        'v_acc': [], 
        't_acc_frame': [], 
        'v_acc_frame': [],
        'val_f1': [], 
        'val_auc': []
    }
    
    # --- MUDANÇA AQUI: Agora rastreamos a Acurácia de VÍDEO ---
    best_val_acc_video = 0.0
    best_val_loss = float('inf') # Começa infinito para qualquer coisa ser menor
    patience_counter = 0
    print("\n" + "="*60)
    print(f"   INICIANDO TREINO ESTABILIZADO ({NUM_EPOCHS} ÉPOCAS)   ")
    print("="*60)


    for ep in range(NUM_EPOCHS):
        # ---------------- FASE 1: TREINO (LOSS) ----------------
        model.train()
    
        t_loss_accum = 0.0
        train_loss_accum = 0.0 
        steps = 0
        
        train_correct_frames = 0
        train_total_frames = 0
        train_correct_videos = 0  
        train_total_videos = 0  
        opt.zero_grad() 
        pbar = tqdm(train_loader, desc=f"Ep {ep+1}/{NUM_EPOCHS} [Train]")

        for i, batch in enumerate(pbar):
            if batch is None: continue
            imgs, lbls, _ = batch 
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

            with autocast():
                out = model(imgs)
                loss = crit(out, lbls) / ACC_STEPS 

            scaler.scale(loss).backward()

            if (i + 1) % ACC_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            t_loss_accum += loss.item() * ACC_STEPS
            pbar.set_postfix({'loss': f"{(loss.item() * ACC_STEPS):.4f}"})
        
        avg_t_loss = t_loss_accum / len(train_loader)

        # ---------------- FASE 2: AUDITORIA (ACC REAL TREINO) ----------------
        # Aqui pegamos os 95%+ que você quer ver
        acc_t_video, acc_t_frame = auditoria_dataset(train_audit_loader, model, DEVICE, "[testando train]")

        # ---------------- FASE 3: VALIDAÇÃO (SUA LÓGICA ORIGINAL) ----------------
        model.eval() 
        val_loss_accum = 0.0
        val_steps = 0
        correct_v_video = 0; total_v_video = 0
        correct_v_frame = 0; total_v_frame = 0
        all_preds_video = []; all_lbls_video = []; all_probs_frame = []; all_labels_frame = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                imgs, lbls, _ = batch
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            
                with autocast():
                    out_logits = model(imgs)
                    loss = crit(out_logits, lbls)
            
                val_loss_accum += loss.item()
                val_steps += 1
                probs = torch.sigmoid(out_logits)
                preds = torch.round(probs)
            
                for k in range(0, len(preds), FACES_PER_VIDEO):
                    if k + FACES_PER_VIDEO > len(preds): break
                    video_preds = preds[k : k + FACES_PER_VIDEO]
                    video_lbl = lbls[k].item() 
                    is_fake_pred = video_preds.sum().item() > (FACES_PER_VIDEO / 2)
                    is_fake_real = (video_lbl == 1)
                
                    if is_fake_pred == is_fake_real:
                        correct_v_video += 1
                    total_v_video += 1
                    all_preds_video.append(1 if is_fake_pred else 0)
                    all_lbls_video.append(1 if is_fake_real else 0)
            
                correct_v_frame += (preds == lbls).sum().item()
                total_v_frame += lbls.numel()
                all_probs_frame.extend(probs.cpu().numpy()); all_labels_frame.extend(lbls.cpu().numpy())

    # ---------------- CÁLCULO DAS MÉTRICAS ----------------
        avg_t_loss = t_loss_accum / len(train_loader) 
    

    # 2. Calcula as médias de VALIDAÇÃO (usando os nomes deste bloco)
        avg_v_loss = val_loss_accum / len(val_loader)
        acc_v_video = correct_v_video / total_v_video if total_v_video > 0 else 0
        acc_v_frame = correct_v_frame / total_v_frame if total_v_frame > 0 else 0

        # --- MÉTRICAS AVANÇADAS (SKLEARN) - USANDO AS LISTAS DE VÍDEO ---
        # Atenção: Usamos all_lbls_video e all_preds_video, não os frames!
        val_f1 = f1_score(all_lbls_video, all_preds_video, zero_division=0)
        val_prec = precision_score(all_lbls_video, all_preds_video, zero_division=0)
        val_rec = recall_score(all_lbls_video, all_preds_video, zero_division=0)
        
        try:
            val_auc = roc_auc_score(all_lbls_video, all_preds_video)
        except:
            val_auc = 0.5 

        # Salva no History (Mantendo suas chaves para o gráfico funcionar)
        history['t_loss'].append(avg_t_loss)
        history['v_loss'].append(avg_v_loss)
        
        history['t_acc_video'].append(acc_t_video)
        history['v_acc'].append(acc_v_video) # A chave do gráfico é 'v_acc'
        
        history['t_acc_frame'].append(acc_t_frame)
        history['v_acc_frame'].append(acc_v_frame)
        
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        # Pega a LR atual
        current_lr = opt.param_groups[0]['lr']

        # --- PRINT OFICIAL ORGANIZADO ---
        print("\n" + "─" * 60)
        print(f" 📈 RESUMO DA ÉPOCA {ep+1:02d} | LR: {current_lr:.6f}")
        print("─" * 60)
        
        # Linha 1: Loss e Acurácia Geral
        print(f" 🟢 TREINO  | Loss: {avg_t_loss:.4f} | Acc Vídeo: {acc_t_video*100:5.2f}% | Acc frames: {acc_t_frame*100:5.2f}%")
        print(f" 🟠 VALIDA  | Loss: {avg_v_loss:.4f} | Acc Vídeo: {acc_v_video*100:5.2f}% | Acc frames: {acc_v_frame*100:5.2f}%")
        print("─" * 60)
        
        # Linha 2: Métricas Detalhadas de Vídeo (Validação)
        print(f" 🎥 VÍDEO (Val) ->  F1: {val_f1:.3f} | Prec: {val_prec:.3f} | Rec: {val_rec:.3f} | AUC: {val_auc:.3f}")
        print("─" * 60 + "\n")


        
        scheduler.step(avg_v_loss)

        try:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'latest.pth'))
        except Exception as e:
            print(f"⚠️ Erro ao salvar latest: {e}")

        # 2. Early Stopping (Baseado APENAS na Loss)
        if avg_v_loss < best_val_loss:
            print(f"📉 Loss caiu para {avg_v_loss:.4f}. (Paciência zerada)")
            best_val_loss = avg_v_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ Loss não caiu. Paciência: {patience_counter}/{PATIENCE}")

        # 3. Salvamento do BEST MODEL (Sua regra: Vídeo > Frame)
        should_save = False

        # Critério A: Bateu recorde de Vídeo? (Prioridade Máxima)
        if acc_v_video > best_val_acc_video:
            print(f"⭐ NOVO RECORDE DE ACC VÍDEO! ({best_val_acc_video*100:.2f}% -> {acc_v_video*100:.2f}%)")
            best_val_acc_video = acc_v_video
            best_val_acc_frame = acc_v_frame # Atualiza frame também para manter o par sincronizado
            should_save = True

        # Critério B: Empatou no Vídeo, mas ganhou nos Frames? (Desempate)
        elif acc_v_video == best_val_acc_video and acc_v_frame > best_val_acc_frame:
            print(f"🎞️ Empate no Vídeo ({acc_v_video*100:.2f}%), mas RECORDE DE FRAMES ({best_val_acc_frame*100:.2f}% -> {acc_v_frame*100:.2f}%)")
            best_val_acc_frame = acc_v_frame # Atualiza o recorde de frames
            should_save = True

        if should_save:
            try:
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
                print("✅ Checkpoint 'best_model.pth' ATUALIZADO (Melhor Performance)!")
            except Exception as e:
                print(f"⚠️ Erro ao salvar best_model: {e}")

        # 4. Parada
        if patience_counter >= PATIENCE:
            print("\n" + "🛑" * 30)
            print(f"EARLY STOPPING: Loss não melhora há {PATIENCE} épocas.")
            print("🛑" * 30)
            break


   # --- GRÁFICO ATUALIZADO (Incluindo Acc Vídeo) ---
    print("\n📈 Gerando plot completo customizado...")
    try:
        # Garante que temos dados para plotar
        epochs = range(1, len(history['t_loss']) + 1)
        
        # Cria a figura com 2 linhas e 2 colunas
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # =======================================================
        # 1. SUPERIOR ESQUERDO: Loss + Acc Vídeo (Eixo Duplo)
        # =======================================================
        ax1 = axs[0, 0]
        ax2 = ax1.twinx() # Cria um segundo eixo Y compartilhado
        
        # Plot da LOSS (Eixo Esquerdo)
        l1 = ax1.plot(epochs, history['t_loss'], 'b-', label='Loss Treino')
        l2 = ax1.plot(epochs, history['v_loss'], 'c-', label='Loss Val')
        ax1.set_ylabel('Loss', color='b')
        ax1.set_xlabel('Épocas')
        
        # Plot da ACC VÍDEO (Eixo Direito)
        # Verifica se existe acc de treino salva, senão plota só val
        t_acc_vid = history.get('t_acc_video', history.get('train_acc_video', []))
        if t_acc_vid:
            l3 = ax2.plot(epochs, t_acc_vid, 'b--', label='Acc Video Treino')
        else:
            l3 = [] # Lista vazia se não tiver
            
        l4 = ax2.plot(epochs, history['v_acc'], 'c--', label='Acc Video Val')
        ax2.set_ylabel('Acurácia (0-1)', color='c')
        
        # Junta as legendas dos dois eixos
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='center right')
        ax1.set_title('Resumo Geral: Loss & Acc Vídeo')
        ax1.grid(True, alpha=0.3)

        # =======================================================
        # 2. SUPERIOR DIREITO: Acc FRAME (Treino vs Val)
        # =======================================================
        # Verifica as chaves corretas do seu dicionário history
        t_acc_frame = history.get('t_acc_frame', history.get('train_acc_frame', []))
        v_acc_frame = history.get('v_acc_frame', history.get('val_acc_frame', []))
        
        if t_acc_frame and v_acc_frame:
            axs[0, 1].plot(epochs, t_acc_frame, 'b-', label='Treino')
            axs[0, 1].plot(epochs, v_acc_frame, 'orange', linestyle='--', label='Validação')
            axs[0, 1].set_title('Acurácia de FRAME')
            axs[0, 1].legend()
            axs[0, 1].grid(True, alpha=0.3)
        else:
            axs[0, 1].text(0.5, 0.5, 'Dados de Acc Frame não encontrados', ha='center')

        # =======================================================
        # 3. INFERIOR ESQUERDO: F1 Score
        # =======================================================
        f1_data = history.get('val_f1', history.get('f1', []))
        if f1_data:
            axs[1, 0].plot(epochs, f1_data, color='green', marker='.', label='F1 Score')
            axs[1, 0].set_title('F1 Score (Validação)')
            axs[1, 0].legend()
            axs[1, 0].grid(True, alpha=0.3)
        else:
            axs[1, 0].text(0.5, 0.5, 'F1 Score não encontrado', ha='center')

        # =======================================================
        # 4. INFERIOR DIREITO: AUC-ROC
        # =======================================================
        auc_data = history.get('val_auc', history.get('auc', []))
        if auc_data:
            axs[1, 1].plot(epochs, auc_data, color='purple', marker='.', label='AUC')
            axs[1, 1].set_title('AUC-ROC (Validação)')
            axs[1, 1].legend()
            axs[1, 1].grid(True, alpha=0.3)
        else:
             axs[1, 1].text(0.5, 0.5, 'AUC não encontrada', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'metrics_full_tcc_v2.png'))
        print("✅ Gráfico 'metrics_full_tcc_v2.png' salvo com sucesso!")
        
    except Exception as e:
        print(f"⚠️ Erro ao plotar gráficos: {e}")
        # Debug: mostre quais chaves tem no history caso dê erro
        print(f"Chaves disponíveis: {history.keys()}")

    print("\n" + "#"*60)
    print("      RELATÓRIO DO EXPERIMENTO - F3-NET / XCEPTION      ")
    print("#"*60)
    
    # Bloco de Treinamento Geral
    print(f"NUM_EPOCHS        = {NUM_EPOCHS}")
    print(f"BATCH_SIZE_VIDEOS = {BATCH_SIZE_VIDEOS}")
    print(f"ACC_STEPS (Grad)  = {ACC_STEPS} (Effective Batch: {BATCH_SIZE_VIDEOS * ACC_STEPS})")
    print(f"FACES_PER_VIDEO   = {FACES_PER_VIDEO}")
    print(f"INPUT_SIZE        = {INPUT_SIZE}")
    print("-" * 60)
    
    # Bloco do Otimizador e Regularização
    print(f"LEARNING_RATE     = {LEARNING_RATE}")
    print(f"MOMENTUM          = {MOMENTUM}")
    print(f"WEIGHT_DECAY      = {WEIGHT_DECAY}")
    print(f"DROPOUT_RATE      = {DROPOUT_RATE}")
    print(f"SCHEDULER FACTOR  = {FACTOR}")
    print(f"PATIENCE (EarlyS) = {PATIENCE}")
    print("-" * 60)

def main_test():
    import warnings
    warnings.filterwarnings("ignore")
    
    # --- CONFIGURAÇÃO ---
    # False usa o 'best_model.pth', que geralmente é o mais estável
    USAR_LATEST = False
    
    print("\n" + "="*60)
    print(f"{'INICIANDO TESTE FINAL - RELATÓRIO COMPLETO':^60}")
    print("="*60)
    
    if not os.path.exists(CACHE_TEST):
        print(f"❌ Pasta de teste não encontrada: {CACHE_TEST}"); return

    # Loader configurado para 1 vídeo por vez para facilitar a votação
    test_ds = VideoFaceDataset(cache_dir=CACHE_TEST,frames_per_video=FACES_PER_VIDEO, mode='test')
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"🚀 Preparando modelo no device: {DEVICE}")
    model = F3Net().to(DEVICE)
    
    arquivo_modelo = 'latest.pth' if USAR_LATEST else 'best_model.pth'
    load_path = os.path.join(SAVE_DIR, arquivo_modelo)
    
    if not os.path.exists(load_path):
        print(f"❌ Modelo {arquivo_modelo} não encontrado."); return

    print(f"✅ Carregando pesos: {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=DEVICE))
    model.eval()

    y_true_video = []
    y_pred_video = []
    video_results = []
    y_score_video = [] # Nova lista para armazenar os scores contínuos

    # Listas para métricas de FRAMES (TUDO JUNTO)
    y_true_frames = []
    y_pred_frames = []

    print("⏳ Rodando inferência em todos os vídeos...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Progresso"):
            imgs, lbls, names = batch
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            
            # --- CORREÇÃO 1: Ajuste de Dimensão (5D para 4D) ---
            # Transforma [1, 40, 3, 299, 299] em [40, 3, 299, 299]
            b, f, c, h, w = imgs.shape
            imgs_flat = imgs.view(-1, c, h, w) 

            # Inferência
            with torch.amp.autocast(device_type='cuda' if 'cuda' in str(DEVICE) else 'cpu'):
                out_flat = model(imgs_flat)
                probs = torch.sigmoid(out_flat)
                preds = torch.round(probs) 


            preds_np = preds.detach().cpu().numpy().flatten().astype(int)
            lbls_np = lbls.view(-1).cpu().numpy().flatten().astype(int)
            
            y_pred_frames.extend(preds_np)
            y_true_frames.extend(lbls_np)

            # Lógica de Votação
            num_fake_frames = preds.sum().item()
            total_frames = f
            
            # Decisão do Vídeo: Maioria Simples
            is_fake_pred = num_fake_frames >= (total_frames / 2)
            
            # --- CORREÇÃO 2: Conversão Segura de Rótulo ---
            # lbls.view(-1)[0] garante que pegamos apenas o primeiro valor do tensor
            is_fake_real = (lbls.view(-1)[0].item() == 1)
            
            y_true_video.append(1 if is_fake_real else 0)
            y_pred_video.append(1 if is_fake_pred else 0)
            y_score_video.append(num_fake_frames / total_frames) # Captura do score para AUC

            video_results.append({
                'name': os.path.basename(names[0])[:35],
                'pred': 'FAKE' if is_fake_pred else 'REAL',
                'real': 'FAKE' if is_fake_real else 'REAL',
                'conf': (num_fake_frames / total_frames) * 100
            })

    # --- RELATÓRIO DE TODOS OS VÍDEOS ---
    print("\n" + "-"*85)
    print(f"{'NOME DO VÍDEO':<35} | {'PRED':<6} | {'REAL':<6} | {'% FRAMES FAKES'}")
    print("-" * 85)
    
    # Exibe todos os resultados sem limitação
    for res in video_results:
        print(f"{res['name']:<35} | {res['pred']:<6} | {res['real']:<6} | {res['conf']:.1f}%")

    print("-" * 85)
    print(f"Total de vídeos testados: {len(video_results)}")

    # --- CÁLCULO DAS MÉTRICAS ---
    acc = accuracy_score(y_true_video, y_pred_video)
    prec = precision_score(y_true_video, y_pred_video, zero_division=0)
    rec = recall_score(y_true_video, y_pred_video, zero_division=0)
    f1 = f1_score(y_true_video, y_pred_video, zero_division=0)
    auc = roc_auc_score(y_true_video, y_score_video)

    y_true_np = np.array(y_true_frames)
    y_pred_np = np.array(y_pred_frames)
    
    total_frames = len(y_true_np)
    correct_frames = np.sum(y_true_np == y_pred_np) # Conta quantos bateram igual
    acc_frames_pct = (correct_frames / total_frames) * 100

    print("\n" + "="*50)
    print(f"🏆 PERFORMANCE FINAL DO MODELO")
    print("="*50)
    print(f"Acurácia Geral: {acc*100:.2f}%")
    print(f"🖼️  Acurácia (FRAMES): {correct_frames}/{total_frames} ({acc_frames_pct:.2f}%)")
    print(f"AUC:            {auc:.4f}")
    print(f"Precisão:       {prec:.4f}")
    print(f"Recall:         {rec:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    print("="*50)

    # --- GERAÇÃO DA MATRIZ DE CONFUSÃO ---
    try:
        cm = confusion_matrix(y_true_video, y_pred_video)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
        disp.plot(cmap='Blues', values_format='d', ax=ax)
        plt.title('Matriz de Confusão - Classificação de Vídeos')
        
        save_path_cm = os.path.join(SAVE_DIR, 'confusion_matrix_final.png')
        plt.savefig(save_path_cm)
        print(f"\n✅ Matriz de confusão salva com sucesso em: {save_path_cm}")
    except Exception as e:
        print(f"⚠️ Erro ao gerar matriz: {e}")

    
    try:
        # 1. Prepara os dados
        df = pd.DataFrame(video_results)

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6)) # Largura ajustada para acomodar mais marcações

        # A MÁGICA: Definindo as bordas exatas das caixas para valores discretos (0, 5, 10... 100)
        bins_exatos = np.arange(-2.5, 105, 5)

        # 2. Cria o Histograma com as duas cores sobrepostas
        ax = sns.histplot(
            data=df, 
            x='conf',      # Eixo X: Porcentagem de Frames Fakes (0 a 100)
            hue='real',    # Separa as cores por Real vs Fake
            element="step", 
            stat="count",
            kde=False,     
            palette={'REAL': 'blue', 'FAKE': 'red'},
            bins=bins_exatos,
            multiple="layer" # Garante que as cores se sobreponham sem somar as alturas
        )

        # Força as marcações do eixo X a irem de 5 em 5 centralizadas nas caixas
        plt.xticks(np.arange(0, 101, 5))
        
        if ax.get_legend() is not None:
            ax.get_legend().set_title("")

        plt.title('Distribuição dos Vídeos por % de Frames Fakes')
        plt.xlabel('Certeza do Modelo')
        plt.ylabel('Quantidade de Vídeos')
        
        # Dá um respiro nas bordas para o 0 e o 100 não cortarem no limite do gráfico
        plt.xlim(-5, 105) 
        plt.ylim(0, 700)
        
        # 3. Salva
        save_path_hist = os.path.join(SAVE_DIR, 'histograma_certeza_videos.png')
        plt.savefig(save_path_hist, dpi=300, bbox_inches='tight') # Adicionado dpi=300 para alta qualidade
        print(f"✅ Histograma de Distribuição salvo em: {save_path_hist}")
        plt.show()

    except Exception as e:
        print(f"⚠️ Erro ao gerar histograma: {e}")
        

if __name__ == '__main__':
    main_train()
    main_test()
