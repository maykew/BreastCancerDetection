import os
import cv2

novo_diretorio = 'dataset_cancer_v1'
caminho_dataset = f"./dataset/BreaKHis_v1/histology_slides/breast"
ampliacoes = ['40X', '100X', '200X', '400X']
diretorios_tarefas_classificacao = {"classificacao_binaria": ["benign", "malignant"], "classificacao_multiclasse": ["adenosis","fibroadenoma","phyllodes_tumor","tubular_adenoma","ductal_carcinoma","lobular_carcinoma","mucinous_carcinoma","papillary_carcinoma"]}
diretorios_imagens = [f"./dataset/BreaKHis_v1/histology_slides/breast/benign", f"./dataset/BreaKHis_v1/histology_slides/breast/malignant"]
imagens = {}

if not os.path.exists(f"./{novo_diretorio}"):
    for tarefa_classificacao, classes in diretorios_tarefas_classificacao.items():
        for ampliacao in ampliacoes:
                for classe in classes:
                    os.makedirs(os.path.join(f"./{novo_diretorio}", tarefa_classificacao, ampliacao, classe))
else:
    print(f'Já existe uma pasta com o nome "{novo_diretorio}" no caminho informado')


for diretorio_imagens in diretorios_imagens:
    for diretorio, subpastas, arquivos in os.walk(diretorio_imagens):
        for arquivo in arquivos:
          if '.png' in arquivo:
            
            partes_caminho_dir = diretorio.split('/')[-1].split('\\')
            tipo_tumor = partes_caminho_dir[0]
            tipo_histologico = partes_caminho_dir[2]
            ampliacao = partes_caminho_dir[-1]
            
            if ampliacao not in imagens: imagens[ampliacao] = {}
            if tipo_tumor not in imagens[ampliacao]: imagens[ampliacao][tipo_tumor] = []
            if tipo_histologico not in imagens[ampliacao]: imagens[ampliacao][tipo_histologico] = []

            imagens[ampliacao][tipo_tumor].append({
                "arquivo": arquivo,
                "caminho": f"{diretorio}/{arquivo}",
                "tipo_tumor": tipo_tumor
            })

            imagens[ampliacao][tipo_histologico].append({
                "arquivo": arquivo,
                "caminho": f"{diretorio}/{arquivo}",
                "tipo_histologico": tipo_histologico
            })

for ampliacao, classes in imagens.items():
    for classe, values in classes.items():
        if classe == 'benign' or classe == 'malignant':
            print(f"{ampliacao} - {classe}")
            for i in range(0, len(values)):
                imagem = cv2.imread(values[i]["caminho"])
                imagem_redimensionada = cv2.resize(imagem, (224, 224))
                classe = values[i]["tipo_tumor"]
                cv2.imwrite(f'./{novo_diretorio}/classificacao_binaria/{ampliacao}/{classe}/{values[i]["arquivo"]}', imagem_redimensionada)
        else:
            print(f"{ampliacao} - {classe}")
            for i in range(0, len(values)):
                imagem = cv2.imread(values[i]["caminho"])
                imagem_redimensionada = cv2.resize(imagem, (224, 224))
                classe = values[i]["tipo_histologico"]
                cv2.imwrite(f'./{novo_diretorio}/classificacao_multiclasse/{ampliacao}/{classe}/{values[i]["arquivo"]}', imagem_redimensionada)


