import cv2
import os
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
import glob
import argparse

# Mapeamento de Strings para constantes do OpenCV
COLOR_MAP = {
    'bgr':   None,                # Padrão do OpenCV
    'rgb':   cv2.COLOR_BGR2RGB,   # Padrão para visualização humana/PIL
    'gray':  cv2.COLOR_BGR2GRAY,  # 1 Canal (Leve)
    'hsv':   cv2.COLOR_BGR2HSV,   # Bom para detectar sombras/saturação
    'ycrcb': cv2.COLOR_BGR2YCrCb, # 🔥 CRUCIAL para Deepfakes (Crominância)
    'lab':   cv2.COLOR_BGR2LAB,
    'yuv':   cv2.COLOR_BGR2YUV,
    'luv':   cv2.COLOR_BGR2LUV
}

def process_video_pipeline(video_path, output_root, num_frames_target, img_size, detector, color_space='bgr', margin_scale=1.2):
    """
    Processa UM vídeo: Amostragem -> Detecção -> Corte -> Resize -> Conversão de Cor -> Save
    """
    
    # Nome do vídeo sem extensão
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    
    # Cria pasta de saída: dataset_final/espaco_cor/video_nome
    # Adicionamos o espaço de cor no caminho para organizar melhor
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames_video == 0:
        return 0

    # --- 1. LÓGICA DE AMOSTRAGEM ---
    extract_all = False
    indices_set = set()

    if num_frames_target == 'full':
        extract_all = True
    else:
        target_count = int(num_frames_target)
        if total_frames_video <= target_count:
            indices_to_grab = np.arange(total_frames_video)
        else:
            indices_to_grab = np.linspace(0, total_frames_video - 1, target_count, dtype=int)
        indices_set = set(indices_to_grab)
    
    frame_idx = 0
    saved_count = 0
    
    # Código de conversão do OpenCV
    conversion_code = COLOR_MAP.get(color_space.lower())

    while cap.isOpened():
        ret, frame = cap.read() # Frame vem em BGR
        if not ret:
            break

        should_process = extract_all or (frame_idx in indices_set)

        if should_process:
            try:
                # MTCNN precisa de RGB para detectar corretamente
                rgb_frame_for_detection = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(rgb_frame_for_detection)

                if detections:
                    # Pega a maior face
                    det = max(detections, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = det['box']
                    
                    # Cálculos do quadrado com margem
                    center_x = x + w / 2
                    center_y = y + h / 2
                    max_dim = max(w, h) * margin_scale
                    
                    x1 = int(max(0, center_x - max_dim / 2))
                    y1 = int(max(0, center_y - max_dim / 2))
                    x2 = int(min(frame.shape[1], center_x + max_dim / 2))
                    y2 = int(min(frame.shape[0], center_y + max_dim / 2))
                    
                    # CORTA NO ORIGINAL (BGR) para preservar dados brutos antes da conversão final
                    face_bgr = frame[y1:y2, x1:x2]

                    if face_bgr.size != 0 and face_bgr.shape[0] > 20:
                        # Redimensionamento (Interpolação de alta qualidade)
                        interp = cv2.INTER_AREA if face_bgr.shape[0] > img_size else cv2.INTER_CUBIC
                        face_resized = cv2.resize(face_bgr, (img_size, img_size), interpolation=interp)
                        
                        # --- 2. CONVERSÃO DE COR ---
                        if color_space.lower() == 'bgr':
                            final_face = face_resized
                        elif conversion_code is not None:
                            final_face = cv2.cvtColor(face_resized, conversion_code)
                        else:
                            # Fallback para BGR se der erro
                            final_face = face_resized

                        # --- 3. SALVAMENTO COM OPENCV ---
                        save_name = f"frame_{saved_count:04d}.jpg"
                        save_path = os.path.join(output_dir, save_name)
                        
                        # cv2.imwrite salva os dados exatos da matriz (excelente para YCrCb/HSV)
                        cv2.imwrite(save_path, final_face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        saved_count += 1
            
            except Exception:
                pass

        frame_idx += 1
        
        if not extract_all and indices_set and frame_idx > max(indices_set):
            break

    cap.release()
    return saved_count

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Extração de Faces Multi-Colorspace")
    
    # Argumentos Obrigatórios
    parser.add_argument('--input', '-i', type=str, required=True, help="Pasta dos vídeos originais")
    parser.add_argument('--output', '-o', type=str, required=True, help="Pasta de saída")
    
    # Argumentos Opcionais com Padrões
    parser.add_argument('--frames', '-f', type=str, default='20', help="Qtd frames (int) ou 'full'")
    parser.add_argument('--size', '-s', type=int, default=224, help="Tamanho da imagem (px)")
    parser.add_argument('--color', '-c', type=str, default='bgr', 
                        help=f"Espaço de cor. Opções: {list(COLOR_MAP.keys())}")
    
    args = parser.parse_args()

    # Validação do espaço de cor
    if args.color.lower() not in COLOR_MAP:
        print(f"❌ Erro: Cor '{args.color}' inválida.")
        print(f"Opções: {list(COLOR_MAP.keys())}")
        return

    # Tratamento da flag 'full'
    if args.frames.lower() == 'full':
        target_frames = 'full'
        msg_frames = "TODOS os frames (Full)"
    else:
        target_frames = int(args.frames)
        msg_frames = f"{target_frames} frames/vídeo"

    if not os.path.exists(args.input):
        print(f"❌ Erro: Entrada '{args.input}' não existe.")
        return

    # Coleta vídeos
    extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm')
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(args.input, ext)))
    
    if not video_files:
        print(f"❌ Zero vídeos encontrados em: {args.input}")
        return

    print(f"{'='*60}")
    print(f"🛠  PIPELINE DE EXTRAÇÃO DE FACES 🛠")
    print(f"{'='*60}")
    print(f"📂 Entrada:  {args.input}")
    print(f"📂 Saída:    {args.output}")
    print(f"🎨 Cor:      {args.color.upper()}")
    print(f"🎞  Frames:   {msg_frames}")
    print(f"📏 Tamanho:  {args.size}x{args.size}")
    print(f"🎥 Vídeos:   {len(video_files)}")
    print(f"{'='*60}\n")

    print("Iniciando MTCNN (pode demorar um pouco na 1ª vez)...")
    try:
        detector = MTCNN()
    except Exception as e:
        print(f"❌ Erro crítico MTCNN: {e}")
        return

    total_faces = 0
    # Loop principal
    for video_path in tqdm(video_files, desc=f"Processando ({args.color.upper()})"):
        count = process_video_pipeline(
            video_path, 
            args.output, 
            num_frames_target=target_frames, 
            img_size=args.size, 
            detector=detector,
            color_space=args.color
        )
        total_faces += count

    print(f"\n✅ Processamento Concluído!")
    print(f"Total de faces extraídas: {total_faces}")
    print(f"Dataset salvo em: {args.output}")

if __name__ == "__main__":
    main()