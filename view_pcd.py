import open3d as o3d
import sys

# Vérifie si un nom de fichier a été donné en argument
if len(sys.argv) < 2:
    print("Usage: python view_pcd.py <chemin_vers_le_fichier.pcd>")
    sys.exit()

# Récupère le nom du fichier depuis l'argument de la ligne de commande
file_path = sys.argv[1]

print(f"Chargement du nuage de points : {file_path}")

try:
    # Lit le fichier .pcd
    pcd = o3d.io.read_point_cloud(file_path)

    # Vérifie si le nuage de points est vide
    if not pcd.has_points():
        print(f"Erreur : Le nuage de points à {file_path} est vide ou n'a pas pu être lu.")
    else:
        print("Affichage du nuage de points. Appuyez sur 'q' dans la fenêtre pour fermer.")
        # Affiche le nuage de points dans une fenêtre interactive
        o3d.visualization.draw_geometries([pcd])

except Exception as e:
    print(f"Une erreur est survenue : {e}")