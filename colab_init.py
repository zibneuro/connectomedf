import os, zipfile, shutil
import gdown

# ==== CONFIG ====
link = "https://drive.google.com/file/d/1ZcYZteVQo16eQMhRH1uYekn62-i3d7UN/view?usp=sharing"   # file or folder share link
zip_path = "demo_data.zip"              # where to save the zip locally
extract_dir = "unzipped_tmp"          # temp extract dir
final_folder_name = "data"  # what to rename the extracted folder to
# ================

def is_folder_link(u: str) -> bool:
    return "/folders/" in u

def is_zip(path: str) -> bool:
    if not os.path.exists(path): return False
    with open(path, "rb") as f:
        return f.read(4) == b"PK\x03\x04"

# Clean up from previous runs (optional)
for p in [zip_path, extract_dir, final_folder_name, "downloaded_folder"]:
    if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
    elif os.path.isfile(p): os.remove(p)

if is_folder_link(link):
    # 1) Folder link: download the folder (not zipped by Drive), then zip it locally
    out_dir = "downloaded_folder"
    gdown.download_folder(link, output=out_dir, quiet=False, use_cookies=False)
    # Make a zip locally so the rest of the pipeline is consistent
    shutil.make_archive(os.path.splitext(zip_path)[0], "zip", out_dir)
else:
    # 1) File link: download the ZIP file robustly
    # fuzzy=True lets you paste the full "…/file/d/<ID>/view" link
    # use_cookies=False avoids login-only downloads
    gdown.download(link, zip_path, quiet=False, fuzzy=True, use_cookies=False)

# 2) Validate it's really a zip (common failure = Drive HTML page)
if not is_zip(zip_path):
    head = ""
    try:
        with open(zip_path, "rb") as f:
            head = f.read(512).decode("utf-8", "ignore").lower()
    except Exception:
        pass
    raise SystemExit(
        "Downloaded file is not a valid ZIP.\n"
        + ("Looks like an HTML page (login/quota/virus-scan warning). "
           "Ensure the link is set to 'Anyone with the link', or use a file (not folder) link. "
           "If it's a folder link, use gdown.download_folder as shown.\n")
        + ("If you see 'too many users have viewed or downloaded this file recently', "
           "ask the owner to make a fresh copy and share that copy.\n")
    )

# 3) Extract
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_dir)

# 4) Rename extracted folder
subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
if len(subdirs) == 1:
    src = os.path.join(extract_dir, subdirs[0])
    os.rename(src, final_folder_name)
    shutil.rmtree(extract_dir, ignore_errors=True)
else:
    # multiple items at root of zip; rename the whole extracted dir
    if os.path.exists(final_folder_name):
        shutil.rmtree(final_folder_name, ignore_errors=True)
    os.rename(extract_dir, final_folder_name)

print("Ready:", final_folder_name)