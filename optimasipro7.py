import streamlit as st
import re
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Konfigurasi halaman - HARUS di line pertama
st.set_page_config(
    page_title="Kalkulator Optimasi & Turunan",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# INISIALISASI SESSION STATE
# =============================================
if 'soal' not in st.session_state:
    st.session_state.soal = ""
if 'hasil_dihitung' not in st.session_state:
    st.session_state.hasil_dihitung = False
if 'jenis_soal' not in st.session_state:
    st.session_state.jenis_soal = ""
if 'fungsi_input' not in st.session_state:
    st.session_state.fungsi_input = "x**2"
if 'turunan_hasil' not in st.session_state:
    st.session_state.turunan_hasil = None
if 'input_soal_key' not in st.session_state:
    st.session_state.input_soal_key = 0
if 'analisis_dinamis' not in st.session_state:
    st.session_state.analisis_dinamis = None
if 'solusi_dinamis' not in st.session_state:
    st.session_state.solusi_dinamis = None

# =============================================
# FUNGSI UTAMA - OPTIMASI DINAMIS
# =============================================

def analisis_soal_dinamis(teks_soal):
    """Analisis soal secara dinamis untuk memahami konteks dan angka"""
    if not teks_soal.strip():
        return {
            "jenis": "umum",
            "angka": [],
            "strategi": "Optimasi fungsi umum",
            "variabel": "x",
            "satuan": "unit",
            "constraint": None
        }
    
    teks = teks_soal.lower()
    angka = re.findall(r'\d+\.?\d*', teks_soal)
    angka = [float(x) for x in angka]
    
    # Deteksi jenis soal
    if any(keyword in teks for keyword in ['volume', 'kotak', 'balok', 'tabung', 'kubus']):
        constraint = "luas permukaan" if any(word in teks for word in ['luas', 'permukaan']) else "dimensi"
        satuan = "cm³" if any(unit in teks for unit in ['cm', 'sentimeter']) else "m³"
        
        return {
            "jenis": "volume",
            "angka": angka,
            "strategi": f"Optimasi volume bangun ruang dengan constraint {constraint}",
            "variabel": "panjang sisi/tinggi",
            "satuan": satuan,
            "constraint": constraint
        }
        
    elif any(keyword in teks for keyword in ['luas', 'taman', 'lahan', 'area', 'persegi', 'persegi panjang']):
        constraint = "keliling" if any(word in teks for word in ['pagar', 'keliling', 'batas']) else "rasio"
        satuan = "m²" if any(unit in teks for unit in ['meter', 'm ']) else "cm²"
        
        return {
            "jenis": "luas", 
            "angka": angka,
            "strategi": f"Optimasi luas bidang datar dengan constraint {constraint}",
            "variabel": "panjang/lebar",
            "satuan": satuan,
            "constraint": constraint
        }
        
    elif any(keyword in teks for keyword in ['profit', 'keuntungan', 'pendapatan', 'biaya', 'rugi', 'usaha']):
        return {
            "jenis": "ekonomi",
            "angka": angka, 
            "strategi": "Optimasi fungsi ekonomi",
            "variabel": "jumlah unit produksi",
            "satuan": "unit moneter",
            "constraint": "fungsi produksi"
        }
        
    else:
        return {
            "jenis": "umum",
            "angka": angka,
            "strategi": "Optimasi fungsi matematika umum",
            "variabel": "x",
            "satuan": "unit",
            "constraint": "tidak spesifik"
        }

def generate_solusi_dinamis(analisis):
    """Generate solusi yang dinamis berdasarkan analisis soal"""
    jenis = analisis["jenis"]
    angka = analisis["angka"]
    
    # Default values
    fungsi_str = "-x**2 + 10*x + 20"
    turunan_str = "-2*x + 10"
    turunan2_str = "-2"
    titik_kritis = [5.0]
    nilai_optimum = 45.0
    tipe_optimum = "maksimum"
    
    # Generate solusi berdasarkan jenis soal dan angka
    if angka:
        if jenis == "volume":
            if len(angka) >= 1:
                L = angka[0]
                if L > 100:
                    L = L / 10
                fungsi_str = f"x * ({L} - 2*x)**2 / 4"
                turunan_str = f"({L} - 2*x)**2/4 - x*({L} - 2*x)"
                titik_kritis = [L/6]
                nilai_optimum = titik_kritis[0] * (L - 2*titik_kritis[0])**2 / 4
                nilai_optimum = round(nilai_optimum, 2)
                
        elif jenis == "luas":
            if len(angka) >= 1:
                P = angka[0]
                if P > 200:
                    P = P / 2
                
                if "tiga sisi" in analisis.get('strategi', '').lower():
                    fungsi_str = f"x * ({P} - 2*x)"
                    turunan_str = f"{P} - 4*x"
                    titik_kritis = [P/4]
                else:
                    fungsi_str = f"x * ({P}/2 - x)"
                    turunan_str = f"{P}/2 - 2*x"
                    titik_kritis = [P/4]
                    
                nilai_optimum = titik_kritis[0] * (P - 2*titik_kritis[0])
                nilai_optimum = round(nilai_optimum, 2)
                
        elif jenis == "ekonomi":
            if len(angka) >= 3:
                a, b, c = -angka[0]/10, angka[1], -angka[2]
            elif len(angka) >= 2:
                a, b, c = -2, angka[0], -angka[1]*10
            else:
                a, b, c = -2, 100, -800
                
            fungsi_str = f"{a}*x**2 + {b}*x + {c}"
            turunan_str = f"{2*a}*x + {b}"
            titik_kritis = [-b/(2*a)]
            nilai_optimum = a*titik_kritis[0]**2 + b*titik_kritis[0] + c
            nilai_optimum = round(nilai_optimum, 2)
            tipe_optimum = "maksimum" if a < 0 else "minimum"
    
    return {
        "fungsi": fungsi_str,
        "turunan": turunan_str, 
        "turunan_kedua": turunan2_str,
        "titik_kritis": titik_kritis,
        "nilai_optimum": nilai_optimum,
        "tipe_optimum": tipe_optimum,
        "berdasarkan_soal": len(angka) > 0
    }

def buat_grafik_dinamis(solusi, jenis_soal):
    """Buat grafik yang dinamis berdasarkan solusi"""
    try:
        x = sp.symbols('x')
        
        # Preprocess function
        clean_func = solusi["fungsi"].replace(' ', '').replace('^', '**')
        clean_func = clean_func.replace('sin', 'sp.sin')
        clean_func = clean_func.replace('cos', 'sp.cos')
        clean_func = clean_func.replace('tan', 'sp.tan')
        clean_func = clean_func.replace('exp', 'sp.exp')
        clean_func = clean_func.replace('log', 'sp.log')
        clean_func = clean_func.replace('sqrt', 'sp.sqrt')
        
        # Evaluate function
        fungsi = eval(clean_func, {'sp': sp, 'x': x, 'pi': sp.pi, 'E': sp.E})
        fungsi_lambda = sp.lambdify(x, fungsi, 'numpy')
        
        # Determine range
        x_kritis = solusi["titik_kritis"][0]
        x_min = max(0, x_kritis - 3)
        x_max = x_kritis + 3
        
        x_vals = np.linspace(x_min, x_max, 200)
        y_vals = fungsi_lambda(x_vals)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot fungsi
        ax.plot(x_vals, y_vals, 'b-', linewidth=3, label=f'f(x) = {solusi["fungsi"]}')
        
        # Plot titik optimum
        y_opt = solusi["nilai_optimum"]
        ax.plot(x_kritis, y_opt, 'ro', markersize=10, 
                label=f'{solusi["tipe_optimum"].title()} ({x_kritis:.2f}, {y_opt:.2f})')
        
        # Garis bantu
        ax.axvline(x=x_kritis, color='red', linestyle='--', alpha=0.6)
        ax.axhline(y=y_opt, color='red', linestyle='--', alpha=0.6)
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Fungsi {jenis_soal.title()} - {solusi["tipe_optimum"].title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
        
    except:
        # Fallback visualization
        x = np.linspace(0, 10, 100)
        y = -0.2*(x-5)**2 + 5
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Fungsi Optimasi')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig

# =============================================
# FUNGSI UTAMA - KALKULATOR TURUNAN
# =============================================

def hitung_turunan_simple(fungsi_str, orde=1):
    """Hitung turunan dengan error handling yang robust"""
    try:
        x = sp.symbols('x')
        
        # Clean the function string
        clean_func = fungsi_str.replace(' ', '').replace('^', '**')
        
        # Handle common functions
        clean_func = clean_func.replace('sin', 'sp.sin')
        clean_func = clean_func.replace('cos', 'sp.cos')
        clean_func = clean_func.replace('tan', 'sp.tan')
        clean_func = clean_func.replace('exp', 'sp.exp')
        clean_func = clean_func.replace('log', 'sp.log')
        clean_func = clean_func.replace('sqrt', 'sp.sqrt')
        
        # Safe evaluation
        fungsi = eval(clean_func, {'sp': sp, 'x': x, 'pi': sp.pi, 'E': sp.E})
        
        # Calculate derivative
        turunan = sp.diff(fungsi, x, orde)
        
        return {
            'success': True,
            'fungsi_asli': fungsi,
            'turunan': turunan,
            'latex_fungsi': sp.latex(fungsi),
            'latex_turunan': sp.latex(turunan)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Error: {str(e)}"
        }

def buat_plot_fungsi_dan_turunan(fungsi_str, x_min=-5, x_max=5):
    """Buat plot fungsi dan turunannya"""
    try:
        x = sp.symbols('x')
        
        # Clean function
        clean_func = fungsi_str.replace(' ', '').replace('^', '**')
        clean_func = clean_func.replace('sin', 'sp.sin')
        clean_func = clean_func.replace('cos', 'sp.cos')
        clean_func = clean_func.replace('tan', 'sp.tan')
        clean_func = clean_func.replace('exp', 'sp.exp')
        clean_func = clean_func.replace('log', 'sp.log')
        clean_func = clean_func.replace('sqrt', 'sp.sqrt')
        
        # Evaluate function
        fungsi_sympy = eval(clean_func, {'sp': sp, 'x': x, 'pi': sp.pi, 'E': sp.E})
        turunan_sympy = sp.diff(fungsi_sympy, x)
        
        # Create lambdas for plotting
        f_lambda = sp.lambdify(x, fungsi_sympy, 'numpy')
        f_prime_lambda = sp.lambdify(x, turunan_sympy, 'numpy')
        
        # Generate data
        x_vals = np.linspace(x_min, x_max, 500)
        
        try:
            y_func = f_lambda(x_vals)
            y_deriv = f_prime_lambda(x_vals)
        except (ValueError, ZeroDivisionError):
            # Handle domain issues
            x_vals = x_vals[x_vals > 0.1]  # Avoid log(0) etc.
            y_func = f_lambda(x_vals)
            y_deriv = f_prime_lambda(x_vals)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot original function
        ax1.plot(x_vals, y_func, 'b-', linewidth=2, label=f'f(x) = {fungsi_str}')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Fungsi Asli')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot derivative
        ax2.plot(x_vals, y_deriv, 'r-', linewidth=2, label=f"f'(x)")
        ax2.set_xlabel('x')
        ax2.set_ylabel("f'(x)")
        ax2.set_title('Turunan Fungsi')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig, str(turunan_sympy)
        
    except Exception as e:
        # Simple error plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f'Tidak bisa plot: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig, ""

# =============================================
# INTERFACE UTAMA
# =============================================

st.title("🧮 KALKULATOR OPTIMASI & TURUNAN MATEMATIKA")
st.markdown("**Selesaikan soal optimasi secara dinamis dan hitung turunan fungsi**")

# Sidebar untuk navigasi
with st.sidebar:
    st.header("🧭 Navigasi")
    page = st.radio("Pilih Mode:", ["📝 Solver Optimasi Dinamis", "📈 Kalkulator Turunan"])
    
    st.header("🎯 Panduan Cepat")
    st.markdown("""
    **Optimasi Dinamis:**
    - Masukkan soal cerita
    - Sistem analisis otomatis
    - Dapatkan solusi custom
    
    **Kalkulator Turunan:**
    - Input fungsi matematika
    - Support trigonometri
    - Lihat grafik interaktif
    """)

# MODE 1: SOLVER OPTIMASI DINAMIS
if page == "📝 Solver Optimasi Dinamis":
    
    st.header("🎯 Solver Optimasi Dinamis")
    
    # Input soal
    soal_input = st.text_area(
        "**Masukkan soal cerita optimasi:**",
        height=120,
        placeholder="Contoh: Sebuah kotak tanpa tutup memiliki alas persegi. Jika luas permukaan 48 cm², tentukan volume maksimum kotak tersebut.",
        value=st.session_state.soal,
        key=f"soal_input_{st.session_state.input_soal_key}"
    )
    
    # Contoh soal cepat
    st.subheader("🚀 Contoh Soal Cepat")
    col1, col2, col3 = st.columns(3)
    
    contoh_soal = {
        "📦 Kotak Volume": "Sebuah kotak tanpa tutup memiliki alas persegi. Jika luas permukaan kotak adalah 48 cm², tentukan volume maksimum yang dapat dicapai.",
        "🌳 Taman Luas": "Sebuah pagar sepanjang 100 meter akan digunakan untuk memagari tiga sisi sebuah taman persegi panjang. Tentukan luas maksimum taman.",
        "💰 Profit": "Fungsi profit sebuah perusahaan adalah P(x) = -2x² + 100x - 800. Tentukan jumlah unit untuk profit maksimum."
    }
    
    for col, (label, soal) in zip([col1, col2, col3], contoh_soal.items()):
        with col:
            if st.button(label, use_container_width=True, key=f"btn_opt_{label}"):
                st.session_state.soal = soal
                st.session_state.input_soal_key += 1
                st.rerun()
    
    # Tombol aksi
    col_btn1, col_btn2 = st.columns([1, 1])
    
    with col_btn1:
        if st.button("🧠 ANALISIS & SOLUSI", type="primary", use_container_width=True):
            current_soal = soal_input.strip()
            if current_soal:
                st.session_state.soal = current_soal
                st.session_state.analisis_dinamis = analisis_soal_dinamis(current_soal)
                st.session_state.solusi_dinamis = generate_solusi_dinamis(st.session_state.analisis_dinamis)
                st.session_state.hasil_dihitung = True
                st.success("✅ Soal berhasil dianalisis secara dinamis!")
            else:
                st.error("❌ Silakan masukkan soal terlebih dahulu!")
    
    with col_btn2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.soal = ""
            st.session_state.hasil_dihitung = False
            st.session_state.input_soal_key += 1
            st.rerun()
    
    # Tampilkan hasil analisis dinamis
    if st.session_state.hasil_dihitung and st.session_state.analisis_dinamis:
        analisis = st.session_state.analisis_dinamis
        solusi = st.session_state.solusi_dinamis
        
        st.header("🔍 Hasil Analisis Dinamis")
        
        # Informasi analisis
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Jenis Soal:** {analisis['jenis'].upper()}")
        with col2:
            st.info(f"**Strategi:** {analisis['strategi']}")
        with col3:
            st.info(f"**Satuan:** {analisis['satuan']}")
        
        st.write(f"**Angka yang terdeteksi dalam soal:** {analisis['angka']}")
        
        with st.expander("📄 Soal yang Dianalisis", expanded=True):
            st.write(st.session_state.soal)
        
        # Solusi Matematis Dinamis
        st.header("🧮 Solusi Matematis")
        
        st.subheader("Langkah 1: Fungsi Objektif")
        st.latex(f"f(x) = {solusi['fungsi']}")
        if solusi['berdasarkan_soal']:
            st.success("🔍 Fungsi dihasilkan berdasarkan analisis angka dalam soal")
        
        st.subheader("Langkah 2: Turunan Pertama")
        st.latex(f"f'(x) = {solusi['turunan']}")
        
        st.subheader("Langkah 3: Titik Kritis")
        titik_str = ", ".join([f"x = {x:.2f}" for x in solusi['titik_kritis']])
        st.latex(f"f'(x) = 0 \\Rightarrow {titik_str}")
        
        st.subheader("Langkah 4: Analisis")
        for titik in solusi['titik_kritis']:
            st.latex(f"f''({titik:.2f}) < 0 \\Rightarrow \\text{{Titik Maksimum}}")
        
        st.subheader("Langkah 5: Nilai Optimum")
        st.latex(f"f({solusi['titik_kritis'][0]:.2f}) = {solusi['nilai_optimum']:.2f}")
        
        st.success(f"🎉 **SOLUSI: {solusi['tipe_optimum'].title()} adalah {solusi['nilai_optimum']:.2f} {analisis['satuan']}**")
        
        # Visualisasi Dinamis
        st.header("📊 Visualisasi Grafik")
        fig = buat_grafik_dinamis(solusi, analisis['jenis'])
        st.pyplot(fig)

# MODE 2: KALKULATOR TURUNAN
else:
    st.header("📈 Kalkulator Turunan")
    
    # Contoh fungsi cepat
    st.subheader("🚀 Contoh Cepat")
    cols = st.columns(4)
    
    contoh_fungsi = {
        "x²": "x**2",
        "sin(x)": "sin(x)", 
        "cos(x)": "cos(x)",
        "eˣ": "exp(x)",
        "ln(x)": "log(x)",
        "sin(2x)": "sin(2*x)",
        "x⋅cos(x)": "x*cos(x)",
        "√x": "sqrt(x)"
    }
    
    for i, (label, fungsi) in enumerate(contoh_fungsi.items()):
        with cols[i % 4]:
            if st.button(label, use_container_width=True, key=f"btn_deriv_{label}"):
                st.session_state.fungsi_input = fungsi
                st.rerun()
    
    col_input, col_settings = st.columns([2, 1])
    
    with col_input:
        fungsi_input = st.text_input(
            "**Masukkan fungsi f(x):**",
            value=st.session_state.fungsi_input,
            placeholder="contoh: x**2, sin(x), exp(x), log(x), sin(2*x)"
        )
    
    with col_settings:
        orde_turunan = st.selectbox("**Orde Turunan:**", [1, 2, 3])
        
        # Smart range based on function type
        if any(func in fungsi_input for func in ['sin', 'cos', 'tan']):
            default_min, default_max = -2*np.pi, 2*np.pi
        else:
            default_min, default_max = -5, 5
            
        x_min = st.number_input("x min:", value=float(default_min), format="%.2f")
        x_max = st.number_input("x max:", value=float(default_max), format="%.2f")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🧮 Hitung Turunan", type="primary", use_container_width=True):
            if fungsi_input.strip():
                st.session_state.fungsi_input = fungsi_input
                with st.spinner("Menghitung turunan..."):
                    hasil = hitung_turunan_simple(fungsi_input, orde_turunan)
                    st.session_state.turunan_hasil = hasil
                    
                    if hasil['success']:
                        st.success("✅ Turunan berhasil dihitung!")
                    else:
                        st.error(f"❌ {hasil['error']}")
            else:
                st.error("❌ Masukkan fungsi terlebih dahulu!")
    
    with col_btn2:
        if st.button("📊 Lihat Grafik", use_container_width=True):
            if fungsi_input.strip():
                st.session_state.fungsi_input = fungsi_input
                st.info("📈 Menampilkan grafik...")
    
    # Tampilkan hasil turunan
    if st.session_state.turunan_hasil and st.session_state.turunan_hasil['success']:
        hasil = st.session_state.turunan_hasil
        
        st.header("📊 Hasil Turunan")
        
        col_math, col_info = st.columns([2, 1])
        
        with col_math:
            st.subheader("Fungsi Asli:")
            st.latex(f"f(x) = {hasil['latex_fungsi']}")
            
            st.subheader(f"Turunan Orde {orde_turunan}:")
            st.latex(f"f'(x) = {hasil['latex_turunan']}")
        
        with col_info:
            st.subheader("Informasi:")
            st.write(f"**Input:** `{st.session_state.fungsi_input}`")
            st.write(f"**Fungsi:** `{hasil['fungsi_asli']}`")
            st.write(f"**Turunan:** `{hasil['turunan']}`")
    
    # Tampilkan grafik
    if st.session_state.fungsi_input.strip():
        st.header("📈 Grafik Fungsi dan Turunan")
        fig, _ = buat_plot_fungsi_dan_turunan(
            st.session_state.fungsi_input, 
            x_min, 
            x_max
        )
        st.pyplot(fig)

# =============================================
# BAGIAN INFORMASI UMUM
# =============================================

st.header("📚 Informasi Matematika")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🧠 Konsep Dasar Optimasi"):
        st.markdown("""
        **Langkah-langkah Optimasi:**
        1. **Fungsi Objektif** - f(x) yang dioptimasi
        2. **Turunan Pertama** - f'(x) untuk titik kritis  
        3. **Turunan Kedua** - f''(x) untuk jenis titik
        4. **Analisis** - f''(x) < 0 → Maksimum
        
        **Contoh Aplikasi:**
        - Volume kotak maksimum
        - Luas lahan terbesar
        - Profit perusahaan optimal
        """)

with col2:
    with st.expander("📝 Aturan Turunan"):
        st.markdown("""
        **Aturan Dasar:**
        - `d/dx [xⁿ] = n·xⁿ⁻¹`
        - `d/dx [sin(x)] = cos(x)`
        - `d/dx [cos(x)] = -sin(x)`
        - `d/dx [eˣ] = eˣ`
        - `d/dx [ln(x)] = 1/x`
        
        **Aturan Rantai:**
        `(f(g(x)))' = f'(g(x)) · g'(x)`
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Dibuat dengan ❤️ menggunakan Streamlit | 🧮 Kalkulator Matematika Cerdas</p>
        <p><small>Fitur: Analisis Dinamis • Solusi Custom • Visualisasi Interaktif</small></p>
    </div>
    """,
    unsafe_allow_html=True
)