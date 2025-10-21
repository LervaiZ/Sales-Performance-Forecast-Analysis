# Sales Performance Forecast & Analysis – OLS on log(Total)

## Overview
วิเคราะห์ยอดขายและสร้างโมเดลเพื่อหาตัวขับเคลื่อน “ยอดต่อบิล (Total)”
- เป้าหมายโมเดล: `log(Total)` เพื่อจัดการสเกล/ความแปรปรวน
- วิธี: Ordinary Least Squares (OLS) + HC3 robust standard errors (กัน heteroskedasticity)
- ฟีเจอร์: Quantity, Unit_price, Rating, City, Product_line, Customer_type, Payment, Month

## Key Results (จากชุดข้อมูลตัวอย่างที่ใช้ในโน้ตบุ๊ก)
- Adj R²: 0.913
- RMSE (log): 0.2718  (~31% multiplicative error)
- RMSE (original): 107.29

### Significant Drivers (p < 0.05)
- Quantity (per +1): ~+25.94%
- Unit_price (per +1): ~+2.17%
- City = Naypyitaw: ~−5.01% vs base

หมายเหตุ: ใช้ HC3 robust SE รองรับ heteroskedasticity; residuals ยังมี nonlinearity/หางเบี้ยวเล็กน้อยแต่ยอมรับได้

### Example Prediction (original scale, 95% CI)
- Predicted Total ≈ 286.82 (95% CI: 270.68 – 303.93)
- เคส: Qty=6, Unit_price=60, City=Naypyitaw, Product_line=Food and beverages, Customer_type=Member, Payment=Credit card, Month=1

## Repository Structure
```
.
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   └── one_click_summary.py        # สคริปต์สร้างรายงาน/รูป/ตารางแบบกดครั้งเดียว
├── assets/
│   └── (diagnostics_resid_qq.png)  # จะถูกสร้างเมื่อรันสคริปต์
├── executive_summary.txt           # บทสรุปสั้น (ตัวอย่างผลลัพธ์)
└── significant_terms.csv           # ตัวแปรสำคัญ (เฉพาะ p<0.05)
```

## Quickstart (Reproduce)
1) เตรียมไฟล์ข้อมูล CSV อย่างน้อยมีคอลัมน์:
   - `Total`, `Quantity`, `Unit price`, `Rating`, `City`, `Product line`, `Customer type`, `Payment`, (`Date` ถ้ามีจะสร้าง `Month` อัตโนมัติ)
2) ติดตั้งไลบรารี
```bash
pip install -r requirements.txt
```
3) รันสรุปแบบกดครั้งเดียว
```bash
python scripts/one_click_summary.py --input path/to/your_sales.csv --outdir .
```
สคริปต์จะสร้าง:
- `executive_summary.txt`
- `significant_terms.csv` (กรองเฉพาะ p<0.05)
- `assets/diagnostics_resid_qq.png`
- `example_prediction.csv`

## Notes
- โมเดลหลีกเลี่ยงตัวแปรอนุพันธ์ของ `Total` (เช่น cogs, tax, gross income) เพื่อกันข้อมูลรั่ว (leakage)
- ถ้ากราฟ residual ยังโค้ง แนะนำลองเพิ่ม interaction `Quantity:Unit_price` หรือเทอมกำลังสองในสคริปต์

## License
เพิ่มไฟล์ LICENSE ตามต้องการ (เช่น MIT/Apache-2.0)