import os
import pandas as pd

def main():
    file_cicids = 'cicids2017_filtered.csv'
    file_lab = 'lab_flow_labeled.csv'
    output_file = 'flow_dataset_merged.csv'

    # 1. Đọc dữ liệu từ hai file CSV
    print(f"1. Đang đọc dữ liệu từ '{file_cicids}' và '{file_lab}'...")
    df_cicids = pd.read_csv(file_cicids)
    df_lab = pd.read_csv(file_lab)

    print(f"   - Kích thước ban đầu {file_cicids}: {df_cicids.shape[0]:,} dòng, {df_cicids.shape[1]} cột")
    print(f"   - Kích thước ban đầu {file_lab}: {df_lab.shape[0]:,} dòng, {df_lab.shape[1]} cột")

    # 2. Làm sạch tên cột: loại bỏ khoảng trắng thừa ở đầu và cuối chuỗi
    df_cicids.columns = df_cicids.columns.str.strip()
    df_lab.columns = df_lab.columns.str.strip()

    # 3. Tìm các cột chung và danh sách cột bị loại bỏ
    common_cols = [c for c in df_cicids.columns if c in df_lab.columns]
    dropped_cicids = [c for c in df_cicids.columns if c not in common_cols]
    dropped_lab = [c for c in df_lab.columns if c not in common_cols]

    print("\n" + "=" * 60)
    print("THÔNG TIN CÁC CỘT")
    print("=" * 60)
    print(f"Số lượng cột chung: {len(common_cols)}")
    print(f"\nDanh sách {len(common_cols)} cột chung:")
    for idx, col in enumerate(common_cols, 1):
        print(f"  {idx:2d}. {col}")

    print(f"\nSố lượng cột bị loại bỏ từ '{file_cicids}': {len(dropped_cicids)}")
    print(f"Danh sách các cột bị loại bỏ từ '{file_cicids}':")
    for idx, col in enumerate(dropped_cicids, 1):
        print(f"  {idx:2d}. {col}")

    print(f"\nSố lượng cột bị loại bỏ từ '{file_lab}': {len(dropped_lab)}")
    print(f"Danh sách các cột bị loại bỏ từ '{file_lab}':")
    for idx, col in enumerate(dropped_lab, 1):
        print(f"  {idx:2d}. {col}")

    # 4. Gộp hai file dựa trên các cột chung
    print("\n" + "=" * 60)
    print("GỘP DỮ LIỆU VÀ XỬ LÝ DỮ LIỆU TRỐNG (NaN)")
    print("=" * 60)
    df_merged = pd.concat([df_cicids[common_cols], df_lab[common_cols]], ignore_index=True)
    total_rows_before = len(df_merged)
    print(f"Tổng số dòng sau khi gộp (trước khi xóa NaN): {total_rows_before:,}")

    # 5. Xóa các dòng chứa giá trị NaN
    df_merged = df_merged.dropna()
    total_rows_after = len(df_merged)
    nan_rows_dropped = total_rows_before - total_rows_after
    print(f"Số dòng bị xóa do chứa NaN: {nan_rows_dropped:,}")
    print(f"Tổng số dòng sau khi làm sạch: {total_rows_after:,}")

    # 6. In ra số lượng dòng theo từng nhãn (Label)
    print("\n" + "=" * 60)
    print("SỐ LƯỢNG DÒNG THEO TỪNG NHÃN (LABEL)")
    print("=" * 60)
    label_counts = df_merged['Label'].value_counts()
    print(label_counts.to_string())

    # 7. Lưu kết quả ra file flow_dataset_merged.csv
    print("\n" + "=" * 60)
    print(f"LƯU DỮ LIỆU RA FILE '{output_file}'...")
    df_merged.to_csv(output_file, index=False)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Đã lưu thành công '{output_file}' ({file_size_mb:.2f} MB).")
    print("=" * 60)

if __name__ == '__main__':
    main()
