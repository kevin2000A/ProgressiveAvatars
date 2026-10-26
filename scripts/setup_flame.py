"""Copy separately licensed FLAME code and assets from an authorized local checkout."""
import argparse
from pathlib import Path
import shutil
p=argparse.ArgumentParser()
p.add_argument('--from-checkout',type=Path,required=True,help='GaussianAvatars checkout with licensed FLAME assets installed')
a=p.parse_args()
source=a.from_checkout.resolve()/'flame_model'
target=Path(__file__).resolve().parents[1]/'flame_model'
for name in ['flame.py','lbs.py']:shutil.copy2(source/name,target/name)
shutil.copytree(source/'assets',target/'assets',dirs_exist_ok=True)
print('Installed separately obtained FLAME dependencies into',target)
