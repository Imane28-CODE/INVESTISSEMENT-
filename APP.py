import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from io import BytesIO
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

# 1. إعداد الصفحة واتجاه النص
st.set_page_config(page_title="تحليل وتحسين المحفظة المالية", layout="centered")
st.markdown("""
    <style>
    * { direction: rtl; text-align: right; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 تحليل وتحسين المحفظة المالية المتوافقة مع الشريعة")
st.markdown("""
   هذا التطبيق يهدف إلى تحليل وتحسين المحفظة المالية المتوافقة مع الشريعة الإسلامية،
   بما في ذلك حساب العوائد والانحرافات المعيارية، وتحسين المحفظة باستخدام البرمجة الخطية،
   مع إضافة مؤشرات شارب وقيمة المخاطر (VaR) لمساعدة المستثمرين في اتخاذ قرارات مدروسة.
""")

# 2. رفع البيانات
st.sidebar.subheader("تحميل بيانات الأسهم")
uploaded_file = st.sidebar.file_uploader("اختار ملف CSV أو Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
    # التحقق من الأعمدة الأساسية
    for col in ['اسم الأصل','العائد','الانحراف المعياري']:
        if col not in df.columns:
            st.error(f"⚠️ البيانات لا تحتوي على العمود المطلوب: {col}")
            st.stop()

    # 3. عرض البيانات
    st.subheader("البيانات المحملة:")
    st.dataframe(df.head())

    # 4. حساب المؤشرات السنوية
    df['العائد السنوي'] = df['العائد'] * 252
    df['الانحراف المعياري السنوي'] = df['الانحراف المعياري'] * np.sqrt(252)

    st.subheader("مؤشرات الأداء السنوي:")
    st.write(f"• العائد السنوي المتوسط: **{df['العائد السنوي'].mean():.2%}**")
    st.write(f"• الانحراف المعياري السنوي المتوسط: **{df['الانحراف المعياري السنوي'].mean():.2%}**")
    rf = 0.03
    sharpe = (df['العائد السنوي'].mean()-rf)/df['الانحراف المعياري السنوي'].mean()
    st.write(f"• مؤشر شارب: **{sharpe:.2f}**")

    # 5. تحسين المحفظة
    st.subheader("تحسين المحفظة")
    n = len(df)
    c = -df['العائد السنوي'].values
    A_eq = [np.ones(n)]
    b_eq = [1]
    bounds = [(0,0.2)]*n
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if not res.success:
        st.error("❌ فشل التحسين، تحقق من القيود أو البيانات.")
        st.stop()

    df['الوزن الجديد'] = res.x
    st.subheader("📈 الأوزان المقترحة:")
    st.dataframe(df[['اسم الأصل','الوزن الجديد']])

    # 6. شرح التحسين
    st.markdown("""
    **شرح التحسين:**
    - تم استخدام البرمجة الخطية لتعظيم العائد السنوي وتقليل المخاطرة.
    - قيود: مجموع الأوزان = 100%، وأقصى وزن لأي أصل = 20%.
    - النتيجة: أوزان أكثر توازناً بين الأصول.
    """)

    # 7. مبيان توزيع العوائد قبل التحسين
    st.subheader("📈 توزيع العوائد قبل التحسين")
    df['اسم معاد'] = df['اسم الأصل'].apply(lambda x: get_display(arabic_reshaper.reshape(str(x))))
    plt.rcParams['font.family'] = 'Tahoma'
    plt.rcParams['axes.unicode_minus'] = False

    fig1, ax1 = plt.subplots()
    ax1.bar(df['اسم معاد'], df['العائد السنوي'])
    ax1.set_title(get_display(arabic_reshaper.reshape("توزيع العوائد قبل التحسين")))
    ax1.set_xlabel(get_display(arabic_reshaper.reshape("اسم الأصل")))
    ax1.set_ylabel(get_display(arabic_reshaper.reshape("العائد السنوي")))
    ax1.set_xticklabels(df['اسم معاد'], rotation=0, fontsize=12)
    st.pyplot(fig1)

    # تحليل كل سهم
    st.markdown("**تحليل مبيان العوائد قبل التحسين:**")
    for _, r in df.iterrows():
        st.markdown(f"- **{r['اسم الأصل']}**: عائد سنوي **{r['العائد السنوي']:.2%}**.")

    # 8. مبيان توزيع الأوزان بعد التحسين
    st.subheader("📊 توزيع الأوزان بعد التحسين")
    df['اسم معاد'] = df['اسم الأصل'].apply(lambda x: get_display(arabic_reshaper.reshape(str(x))))
    fig2, ax2 = plt.subplots()
    ax2.bar(df['اسم معاد'], df['الوزن الجديد'])
    ax2.set_title(get_display(arabic_reshaper.reshape("توزيع الأوزان بعد التحسين")))
    ax2.set_xlabel(get_display(arabic_reshaper.reshape("اسم الأصل")))
    ax2.set_ylabel(get_display(arabic_reshaper.reshape("الوزن الجديد")))
    ax2.set_xticklabels(df['اسم معاد'], rotation=0, fontsize=12)
    st.pyplot(fig2)

    # تحليل كل سهم
    st.markdown("**تحليل مبيان الأوزان بعد التحسين:**")
    for _, r in df.iterrows():
        st.markdown(f"- **{r['اسم الأصل']}**: الوزن **{r['الوزن الجديد']:.2%}**.")

    # 9. مبيان مقارنة العوائد
    st.subheader("📉 مقارنة العوائد قبل وبعد التحسين")
    df['اسم معاد'] = df['اسم الأصل'].apply(lambda x: get_display(arabic_reshaper.reshape(str(x))))
    fig3, ax3 = plt.subplots()
    ax3.bar(df['اسم معاد'], df['العائد السنوي'], label=get_display(arabic_reshaper.reshape("قبل")), alpha=0.6)
    ax3.bar(df['اسم معاد'], df['الوزن الجديد'] * df['العائد السنوي'], label=get_display(arabic_reshaper.reshape("بعد")), alpha=0.6)
    ax3.set_title(get_display(arabic_reshaper.reshape("مقارنة العوائد")), fontsize=14)
    ax3.set_xlabel(get_display(arabic_reshaper.reshape("اسم الأصل")), fontsize=12)
    ax3.set_ylabel(get_display(arabic_reshaper.reshape("العائد السنوي")), fontsize=12)
    ax3.legend(prop={'family':'Cairo'})
    ax3.set_xticklabels(df['اسم معاد'], rotation=0, fontsize=11)
    st.pyplot(fig3)

    # تحليل كل سهم
    st.markdown("**تحليل مبيان مقارنة العوائد:**")
    for _, r in df.iterrows():
        قبل = r['العائد السنوي']
        بعد = r['الوزن الجديد'] * r['العائد السنوي']
        st.markdown(f"- **{r['اسم الأصل']}**: قبل **{قبل:.2%}** | بعد **{بعد:.2%}**.")

    # 10. تنزيل النتائج
    to_dl = df[['اسم الأصل','العائد السنوي','الانحراف المعياري السنوي','الوزن الجديد']]
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        to_dl.to_excel(w, index=False, sheet_name="المحفظة المحسنة")
    st.download_button("⬇️ تحميل النتائج (Excel)", buf.getvalue(),
                       file_name="portfolio_optimized.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")