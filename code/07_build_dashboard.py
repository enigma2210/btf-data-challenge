import json, sys, shutil, os
repo = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/enigma2210/btf-data-challenge"
tpl = open("dashboard/template.html", encoding="utf-8").read()
data = open("docs/data.json", encoding="utf-8").read()
html = tpl.replace("__DATA__", data).replace("__REPO__", json.dumps(repo))
open("docs/index.html", "w", encoding="utf-8").write(html)
os.makedirs("docs/fonts", exist_ok=True)
src = "/usr/share/texmf/fonts/opentype/public/tex-gyre/"
for f in ["texgyrepagella-regular.otf", "texgyrepagella-bold.otf", "texgyreheros-regular.otf", "texgyreheros-bold.otf", "texgyreheroscn-bold.otf"]:
    if os.path.exists(src + f): shutil.copy(src + f, "docs/fonts/" + f)
print("wrote docs/index.html", len(html))
