"""
    File 처리 utils function
"""
import os
import json
import shutil

def process_reports(src, dst):
    if not os.path.exists(src):
        os.makedirs(src)

    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")
                        continue

                # 타겟 파일의 MD5 해시 추출
                md5_hash = data.get('target', {}).get('file', {}).get('md5', '')

                if md5_hash:
                    new_file_name = f"{md5_hash}.json"
                    new_file_path = os.path.join(dst, new_file_name)
                    shutil.copy(file_path, new_file_path)
                    print(f"Copied and renamed: {file_path} -> {new_file_path}")
                else:
                    print(f"MD5 hash not found in {file_path}")




def main():
    # 사용 예시
    src1 = "../../dataset/reports/benign_reports1"
    src2 = "../../dataset/reports/benign_reports2"
    src3 = "../../dataset/reports/benign_reports3"
    dst = "../../dataset/reports/benign"
    process_reports(src1, dst)
    process_reports(src2, dst)
    process_reports(src3, dst)


if __name__ == "__main__":
    main()