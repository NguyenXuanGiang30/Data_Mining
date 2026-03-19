"""
report.py - Tổng hợp bảng/biểu đồ kết quả

Module này chịu trách nhiệm:
- Tổng hợp kết quả từ tất cả các bước
- Xuất bảng CSV
- Tạo báo cáo markdown
"""

import pandas as pd
import os


def save_results_table(results_df, name, output_dir="outputs/tables"):
    """Lưu bảng kết quả ra CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.csv")
    results_df.to_csv(path)
    print(f"💾 Saved table: {path}")
    return path


def save_rules_table(rules_df, name="association_rules", output_dir="outputs/tables"):
    """Lưu bảng luật kết hợp."""
    return save_results_table(rules_df, name, output_dir)


def create_summary_report(
    eda_stats=None,
    association_rules_count=None,
    clustering_results=None,
    classification_results=None,
    semi_supervised_results=None,
    output_dir="outputs/tables"
):
    """
    Tạo báo cáo tổng hợp.
    
    Returns
    -------
    str
        Nội dung báo cáo dạng text.
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("📋 BÁO CÁO TỔNG HỢP KẾT QUẢ")
    report_lines.append("=" * 60)
    
    if eda_stats:
        report_lines.append(f"\n1. DỮ LIỆU:")
        report_lines.append(f"   - Số mẫu: {eda_stats.get('n_rows', 'N/A')}")
        report_lines.append(f"   - Số features: {eda_stats.get('n_cols', 'N/A')}")
        report_lines.append(f"   - Missing values: {eda_stats.get('total_missing', 'N/A')}")
    
    if association_rules_count:
        report_lines.append(f"\n2. LUẬT KẾT HỢP:")
        report_lines.append(f"   - Tổng số luật: {association_rules_count}")
    
    if clustering_results:
        report_lines.append(f"\n3. PHÂN CỤM:")
        report_lines.append(f"   - Số cụm tối ưu: {clustering_results.get('best_k', 'N/A')}")
        report_lines.append(f"   - Silhouette Score: {clustering_results.get('silhouette', 'N/A'):.4f}")
    
    if classification_results is not None:
        report_lines.append(f"\n4. PHÂN LỚP:")
        report_lines.append(classification_results.to_string())
    
    if semi_supervised_results is not None:
        report_lines.append(f"\n5. BÁN GIÁM SÁT:")
        report_lines.append(semi_supervised_results.to_string())
    
    report = "\n".join(report_lines)
    
    # Lưu file
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"💾 Saved report: {path}")
    
    return report
