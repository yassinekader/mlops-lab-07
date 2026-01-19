# Étape 1 : Préparer l’environnement Kubernetes
<img width="970" height="455" alt="2026-01-19-163111_hyprshot" src="https://github.com/user-attachments/assets/f05a3ea8-be01-4589-9e67-b1d14f81e1fb" />
<img width="589" height="163" alt="2026-01-19-163129_hyprshot" src="https://github.com/user-attachments/assets/0daf3a85-2988-43a4-84bc-f83d0f42484e" />
<img width="640" height="151" alt="2026-01-19-163144_hyprshot" src="https://github.com/user-attachments/assets/b5f23430-e500-4592-9101-66e82318618a" />
<img width="619" height="317" alt="2026-01-19-163210_hyprshot" src="https://github.com/user-attachments/assets/a94e14a7-02e1-4950-b8dd-8e259c5eea03" />

# Étape 2 : Préparer l’image Docker de l’API churn
<img width="480" height="81" alt="2026-01-19-163236_hyprshot" src="https://github.com/user-attachments/assets/d7c91bdf-3660-455a-9546-0ec8a10b5d3f" />
<img width="489" height="59" alt="2026-01-19-163326_hyprshot" src="https://github.com/user-attachments/assets/97c23fc0-8cdc-48f2-81a9-a0891257de8c" />
<img width="927" height="439" alt="2026-01-19-163416_hyprshot" src="https://github.com/user-attachments/assets/a268cf85-397f-4ea6-8a33-d8cd2ae5912b" />

# Étape 3 : Créer le dossier des manifests Kubernetes
<img width="660" height="511" alt="2026-01-19-163713_hyprshot" src="https://github.com/user-attachments/assets/00fadb49-1cde-4e26-881a-0dd5f2cb4723" />

# Étape 4 : Construire l’image Docker (tag versionné)
<img width="1904" height="599" alt="2026-01-19-164005_hyprshot" src="https://github.com/user-attachments/assets/6bca795f-d10e-4782-965c-dcae5338d11b" />
<img width="984" height="111" alt="2026-01-19-164032_hyprshot" src="https://github.com/user-attachments/assets/144864b1-0a7d-4bab-bfad-15ee92fed5d8" />

# Étape 5 : Charger explicitement l’image dans Minikube
<img width="642" height="258" alt="2026-01-19-164213_hyprshot" src="https://github.com/user-attachments/assets/0e25ad24-c9ca-4230-a878-e5db81e53a99" />

# Étape 6 : Deployment Kubernetes pour l’API churn
<img width="938" height="513" alt="2026-01-19-164713_hyprshot" src="https://github.com/user-attachments/assets/85a2b5d1-0034-447d-9a51-1d8d0a971232" />

# Étape 7 : Exposer l’API via un Service NodePort
<img width="781" height="713" alt="2026-01-19-164950_hyprshot" src="https://github.com/user-attachments/assets/7a5f0593-4a9e-4376-8b80-3f9ba151387d" />
<img width="1138" height="490" alt="2026-01-19-165328_hyprshot" src="https://github.com/user-attachments/assets/8b7de7be-cc61-4853-829d-483fe4c5dcbd" />
<img width="1153" height="441" alt="2026-01-19-165409_hyprshot" src="https://github.com/user-attachments/assets/f8733d73-9389-4bf4-a23f-8bcc342df97e" />

# Étape 8 : Injecter la configuration MLOps via ConfigMap
<img width="565" height="756" alt="2026-01-19-165629_hyprshot" src="https://github.com/user-attachments/assets/9737eb85-3d8d-46f9-b602-9e6a6bb81b88" />
<img width="751" height="477" alt="2026-01-19-171153_hyprshot" src="https://github.com/user-attachments/assets/4bcc0399-e0cf-40e5-bc57-3d0e926bc092" />

# Étape 9 : Gérer les secrets (MONITORING_TOKEN) 
<img width="529" height="648" alt="2026-01-19-171329_hyprshot" src="https://github.com/user-attachments/assets/dbb6137f-dc65-47fc-b007-df45dfd2bcb1" />
<img width="732" height="330" alt="2026-01-19-171515_hyprshot" src="https://github.com/user-attachments/assets/d9fdf7a3-bccf-4af0-8a99-fcea73a813ce" />

# Étape 10 : Mise en place des endpoints de santé et des probes Kubernetes pour l’API Churn
<img width="946" height="583" alt="2026-01-19-172301_hyprshot" src="https://github.com/user-attachments/assets/8eed0d91-2476-4f13-b44d-6dd0bc93b991" />

# Étape 11 : Ajouter les probes (liveness / readiness / startup)
<img width="941" height="1031" alt="2026-01-19-172747_hyprshot" src="https://github.com/user-attachments/assets/0676fb4a-f937-4681-a9f2-57597a27b53d" />
<img width="928" height="395" alt="2026-01-19-172847_hyprshot" src="https://github.com/user-attachments/assets/dad45281-83f6-443d-b3bd-304f711deb5e" />

# Étape 12 : Volume persistant pour registry + logs
<img width="951" height="810" alt="2026-01-19-173700_hyprshot" src="https://github.com/user-attachments/assets/8826e4f3-2267-4f66-aecb-95969383740b" />
<img width="956" height="935" alt="2026-01-19-173726_hyprshot" src="https://github.com/user-attachments/assets/b91fd8de-740e-4a35-b9bc-71b0df5d6f24" />

# Étape 13 : NetworkPolicy
<img width="962" height="358" alt="2026-01-19-173836_hyprshot" src="https://github.com/user-attachments/assets/fe05aff1-4abf-4011-955a-85919d399025" />

# Étape 14 : Vérifications finales
<img width="824" height="485" alt="2026-01-19-174025_hyprshot" src="https://github.com/user-attachments/assets/34f5f29a-45e1-43c0-9ab4-48ece93e8a23" />
<img width="790" height="101" alt="2026-01-19-173931_hyprshot" src="https://github.com/user-attachments/assets/5461d912-541f-4e9f-b843-e4c9d9103555" />
<img width="861" height="310" alt="2026-01-19-174320_hyprshot" src="https://github.com/user-attachments/assets/d1d72f68-a6f4-4c29-b4ce-0bdcaddf3d37" />
