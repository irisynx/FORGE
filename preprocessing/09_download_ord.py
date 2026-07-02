import os
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time
import random

LOCAL_DATA_DIR = "./ord-data/data"

GITHUB_TOKEN = ""

MAX_WORKERS = 4

def get_all_pb_files():
    """获取文件列表"""
    print("📡 正在获取文件列表...")
    api_url = "https://api.github.com/repos/Open-Reaction-Database/ord-data/git/trees/main?recursive=1"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code != 200:
            print(f"❌ API 失败: {response.status_code} (可能是 API 限流，请填 Token)")
            return []

        data = response.json()

        files = [
            i["path"]
            for i in data.get("tree", [])
            if i["path"].startswith("data/") and i["path"].endswith(".pb.gz")
        ]
        print(f"✅ 找到 {len(files)} 个文件。")
        return files
    except Exception as e:
        print(f"❌ 错误: {e}")
        return []

def is_fake_lfs_file(filepath):
    """检测是否是假的 LFS 指针文件"""
    if not os.path.exists(filepath):
        return True

    sz = os.path.getsize(filepath)

    if sz < 2048:
        try:
            with open(filepath, "rb") as f:
                head = f.read(50)

                if head.startswith(b"version https://git-lfs"):
                    return True

                if b"<!DOCTYPE html" in head:
                    return True
        except:
            return True
    return False

def download_file(file_path):
    local_path = os.path.join(LOCAL_DATA_DIR, file_path.replace("data/", ""))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not is_fake_lfs_file(local_path):
        return "skipped"

    if os.path.exists(local_path):
        print(f"🛠️ 修复 LFS 指针: {os.path.basename(file_path)}")

    url = f"https://github.com/Open-Reaction-Database/ord-data/raw/main/{file_path}"

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    for attempt in range(5):
        try:

            with requests.get(
                url, headers=headers, stream=True, allow_redirects=True, timeout=120
            ) as r:
                if r.status_code == 404:
                    return "failed_404"
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue

                r.raise_for_status()

                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)

            if is_fake_lfs_file(local_path):

                raise ValueError("下载结果依然是 LFS 指针 (可能是带宽限制)")

            return "downloaded"

        except Exception as e:
            if attempt == 4:
                print(f"❌ {os.path.basename(file_path)} 失败: {e}")
                return "failed"
            time.sleep(2)

    return "failed"

def main():
    files = get_all_pb_files()
    if not files:
        return

    print(f"🚀 开始修复下载 (线程: {MAX_WORKERS})...")
    if not GITHUB_TOKEN:
        print("⚠️  警告: 未检测到 Token。如果下载大量失败，请务必申请并填入 Token！")

    stats = {"ok": 0, "skip": 0, "fail": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_file, f): f for f in files}

        total = len(files)
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            fname = os.path.basename(futures[fut])
            try:
                res = fut.result()
                if res == "downloaded":
                    stats["ok"] += 1
                    print(f"[{i+1}/{total}] ✅ 修复成功: {fname}")
                elif res == "skipped":
                    stats["skip"] += 1
                    if stats["skip"] % 100 == 0:
                        print(f"[{i+1}/{total}] ⏭️  跳过正常文件...")
                else:
                    stats["fail"] += 1
                    print(f"[{i+1}/{total}] ❌ 修复失败: {fname}")
            except:
                stats["fail"] += 1

    print("\n" + "=" * 30)
    print(f"修复: {stats['ok']} | 跳过: {stats['skip']} | 失败: {stats['fail']}")
    print("=" * 30)
    if stats["fail"] > 0:
        print("💡 建议: 如果仍有失败，请安装 git-lfs 并在命令行运行: git lfs pull")

if __name__ == "__main__":
    main()
