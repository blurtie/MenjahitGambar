import streamlit as st
import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Stitching Pipeline", layout="wide")

def main():
    # --- SIDEBAR: KONTROL PARAMETER ---
    st.sidebar.title("🛠️ Kontrol Parameter")
    st.sidebar.markdown("**Tim Pengembang:**\n- Javier Shaw\n- Evan Jonathan Tjahjadi\n- Muhammad Faiqi Harumantaka\n- Darvesh Azraf Fakhiri")
    st.sidebar.markdown("---")
    
    ratio_thresh = st.sidebar.slider("Lowe's Ratio Test", 0.5, 0.9, 0.75, 0.05, help="Ambang batas untuk menyaring kecocokan fitur.")
    ransac_thresh = st.sidebar.slider("RANSAC Threshold (px)", 1.0, 10.0, 5.0, 0.5, help="Toleransi reprojection error untuk inliers.")
    
    st.sidebar.markdown("---")

    # --- JUDUL UTAMA ---
    st.title("Algoritma Image Stitching: Step-by-Step")
    st.markdown("Eksplorasi iteratif dari *pipeline* SIFT + RANSAC berhadapan dengan Baseline OpenCV.")

    # --- AREA UPLOAD ---
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        upload1 = st.file_uploader("Upload Gambar Kiri", type=['jpg', 'jpeg', 'png'])
    with col_up2:
        upload2 = st.file_uploader("Upload Gambar Kanan", type=['jpg', 'jpeg', 'png'])

    if upload1 and upload2:
        img1_bgr = cv2.imdecode(np.asarray(bytearray(upload1.read()), dtype=np.uint8), 1)
        img2_bgr = cv2.imdecode(np.asarray(bytearray(upload2.read()), dtype=np.uint8), 1)

        img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
        img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

        # --- MEMBUAT TABULASI UNTUK SETIAP STEP ---
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Tahap 1: Deteksi (SIFT)", 
            "📐 Tahap 2: Homografi (RANSAC)", 
            "🎨 Tahap 3: Warping & Blending", 
            "📊 Tahap 4: Metrik Evaluasi",
            "🏆 Tahap 5: Komparasi OpenCV"
        ])

        # ==========================================
        # KOMPUTASI GLOBAL (Agar tidak diulang per tab)
        # ==========================================
        with st.spinner('Menghitung geometri komputasional...'):
            start_manual = time.time()
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1_gray, None)
            kp2, des2 = sift.detectAndCompute(img2_gray, None)

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

            manual_success = False
            if len(good_matches) >= 4:
                src_pts = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
                dst_pts = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
                matches_mask = mask.ravel().tolist()
                
                # --- LOGIKA KANVAS STATIS (NAIF) ---
                h1, w1 = img1_rgb.shape[:2]
                h2, w2 = img2_rgb.shape[:2]
                width_panorama = w1 + w2
                height_panorama = max(h1, h2) + int(abs(H[1, 2]))

                # Warping langsung tanpa mempertimbangkan koordinat negatif
                panorama_img2 = cv2.warpPerspective(img2_rgb, H, (width_panorama, height_panorama))
                panorama_img1 = np.zeros((height_panorama, width_panorama, 3), dtype=np.uint8)
                panorama_img1[0:h1, 0:w1] = img1_rgb

                overlap_start = max(0, int(H[0, 2]))
                overlap_end = w1
                
                # Alpha Blending Linear
                alpha_mask = np.zeros((height_panorama, width_panorama, 1), dtype=np.float32)
                alpha_mask[:, :overlap_start] = 1.0
                if (overlap_end - overlap_start) > 0:
                    alpha_mask[:h1, overlap_start:overlap_end] = np.tile(
                        np.linspace(1.0, 0.0, overlap_end - overlap_start).reshape(1, -1, 1), (h1, 1, 1)
                    )

                mask1 = panorama_img1 > 0
                mask2 = panorama_img2 > 0
                both_exist = mask1 & mask2

                panorama_blended = panorama_img2.copy()
                pure_img1_area = mask1 & (~mask2)
                panorama_blended[pure_img1_area] = panorama_img1[pure_img1_area]
                
                panorama_blended[both_exist] = ((panorama_img1.astype(np.float32) * alpha_mask) + 
                                                (panorama_img2.astype(np.float32) * (1.0 - alpha_mask)))[both_exist].astype(np.uint8)

                crop_width = min(width_panorama, w2 + int(H[0, 2]))
                manual_result = panorama_blended[:min(h1, h2), :crop_width]
                
                # Evaluasi SSIM pada koordinat statis
                score_ssim = 0.0
                diff_map = np.empty((0, 0), dtype=np.float32) # Inisialisasi awal untuk Pylance
                
                if overlap_start < overlap_end:
                    gray1 = cv2.cvtColor(img1_rgb[:, overlap_start:overlap_end], cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(panorama_img2[:h1, overlap_start:overlap_end], cv2.COLOR_RGB2GRAY)
                    
                    # Ekstrak manual untuk membungkam type-checker Pylance
                    ssim_out = ssim(gray1, gray2, full=True, data_range=255)
                    score_ssim = float(ssim_out[0])
                    diff_map = ssim_out[1]
                
                manual_success = True
            time_manual = time.time() - start_manual

            # OpenCV Baseline
            start_baseline = time.time()
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, baseline_result = stitcher.stitch([img1_rgb, img2_rgb])
            time_baseline = time.time() - start_baseline


        # ==========================================
        # TAB 1: SIFT DETECT & MATCH
        # ==========================================
        with tab1:
            st.header("Pencocokan Fitur Lokal (SIFT)")
            st.markdown("Algoritma mencari titik ekstrem yang stabil terhadap skala dan rotasi, lalu mencocokkannya menggunakan jarak Euclidean.")
            
            col_t1a, col_t1b = st.columns(2)
            with col_t1a:
                st.metric("Total Keypoints Gbr 1", len(kp1))
            with col_t1b:
                st.metric("Total Keypoints Gbr 2", len(kp2))
            
            st.markdown(f"**Total Matches (Lolos Ratio Test {ratio_thresh}): {len(good_matches)} titik.**")
            img_matches = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            st.image(img_matches, caption="Garis penghubung kecocokan fitur mentah", use_container_width=True)

        if not manual_success:
            st.error("Titik kecocokan kurang dari 4. Tidak bisa melanjutkan ke tahap berikutnya.")
            return

        # ==========================================
        # TAB 2: RANSAC HOMOGRAPHY
        # ==========================================
        with tab2:
            st.header("Estimasi Matriks Homografi")
            st.markdown("RANSAC mengeliminasi kecocokan fitur yang menyilang (outliers) untuk membangun matriks transformasi 2D murni.")
            
            inliers_count = sum(matches_mask)
            col_t2a, col_t2b = st.columns([1, 2])
            with col_t2a:
                st.metric("Valid Inliers", f"{inliers_count} ({inliers_count/len(good_matches)*100:.1f}%)")
                st.write("**Matriks H:**")
                st.dataframe(np.round(H, 4), use_container_width=True)
            
            with col_t2b:
                draw_params = dict(matchColor=(0, 255, 0), matchesMask=matches_mask, flags=2)
                img_inliers = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, **draw_params)
                st.image(img_inliers, caption="Inliers hijau yang lolos batas toleransi RANSAC", use_container_width=True)

        # ==========================================
        # TAB 3: WARPING & BLENDING
        # ==========================================
        with tab3:
            st.header("Hasil Panorama SIFT + RANSAC")
            st.markdown("Menggunakan matriks H untuk melengkungkan Gambar 2, lalu digabung dengan Gambar 1 menggunakan **Linear Alpha Blending**.")
            
            # Kolom di tengah untuk tampilan estetik
            _, col_t3, _ = st.columns([1, 10, 1])
            with col_t3:
                st.image(manual_result, caption="Hasil Akhir Implementasi Manual", use_container_width=True)
                
                is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(manual_result, cv2.COLOR_RGB2BGR))
                if is_success:
                    st.download_button("💾 Unduh Panorama Manual", data=buffer.tobytes(), file_name="manual_panorama.jpg", mime="image/jpeg", use_container_width=True)

        # ==========================================
        # TAB 4: EVALUASI METRIK
        # ==========================================
        with tab4:
            st.header("Analisis Evaluasi Objektif")
            col_t4a, col_t4b = st.columns(2)
            
            with col_t4a:
                st.subheader("1. Reprojection Error")
                inlier_src = src_pts[mask.ravel() == 1]
                inlier_dst = dst_pts[mask.ravel() == 1]
                errors = np.sqrt(np.sum((inlier_src - cv2.perspectiveTransform(inlier_dst, H))**2, axis=2))
                mean_err = np.mean(errors)
                st.metric("Mean Error", f"{mean_err:.4f} Piksel")
                
                fig_hist, ax_hist = plt.subplots(figsize=(6,4))
                ax_hist.hist(errors.ravel(), bins=20, color='skyblue', edgecolor='black')
                ax_hist.set_title("Distribusi Error Inliers")
                ax_hist.set_xlabel("Error (Piksel)")
                ax_hist.set_ylabel("Frekuensi")
                st.pyplot(fig_hist)
                
            with col_t4b:
                st.subheader("2. Structural Similarity Index")
                st.metric("SSIM Area Overlap", f"{score_ssim:.4f}")
                
                if score_ssim > 0:
                    fig_ssim, ax_ssim = plt.subplots(figsize=(6, 4))
                    cax = ax_ssim.imshow(np.array(diff_map * 255, dtype=np.uint8), cmap='hot')
                    fig_ssim.colorbar(cax, label="Kesamaan (Terang = Presisi)")
                    ax_ssim.axis('off')
                    ax_ssim.set_title("Peta Perbedaan Struktur")
                    st.pyplot(fig_ssim)

        # ==========================================
        # TAB 5: KOMPARASI OPENCV
        # ==========================================
        with tab5:
            st.header("Baseline vs Manual")
            col_t5a, col_t5b = st.columns(2)
            
            with col_t5a:
                st.markdown("### Metode Manual (SIFT+RANSAC)")
                st.image(manual_result, use_container_width=True)
                st.info(f"⏱️ Waktu Eksekusi: {time_manual:.3f} detik")

            with col_t5b:
                st.markdown("### OpenCV Built-in (PANORAMA Mode)")
                if status == cv2.Stitcher_OK:
                    st.image(baseline_result, use_container_width=True)
                    st.info(f"⏱️ Waktu Eksekusi: {time_baseline:.3f} detik")
                    
                    is_success_cv, buffer_cv = cv2.imencode(".jpg", cv2.cvtColor(baseline_result, cv2.COLOR_RGB2BGR))
                    if is_success_cv:
                        st.download_button("💾 Unduh Panorama OpenCV", data=buffer_cv.tobytes(), file_name="opencv_panorama.jpg", mime="image/jpeg", use_container_width=True)
                elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
                    st.error("⚠️ OpenCV Stitcher menolak menggabungkan gambar (Error 1).")
                    st.markdown("""
                    **Analisis Teknis:**
                    Algoritma *Bundle Adjustment* OpenCV gagal merekonstruksi model kamera 3D secara global. Hal ini umumnya terjadi karena:
                    1. **Paralaks Ekstrem:** Translasi kamera pada objek yang sangat dekat.
                    2. **Objek Dinamis:** Pergerakan kecil pada objek *foreground* (seperti manusia) yang membuat vektor geometri saling bertentangan.
                    
                    *Ini membuktikan bahwa pendekatan SIFT+RANSAC manual lebih kebal (robust) untuk memaksa penggabungan pada gambar kasual non-ideal, meskipun mengorbankan sedikit akurasi optik.*
                    """)

if __name__ == '__main__':
    main()