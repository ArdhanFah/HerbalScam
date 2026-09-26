/* ═══════════════════════════════════════════════
   HerbalScan — Main JavaScript
   ═══════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    initNavbar();
    initScrollAnimations();
    initCounterAnimations();
    initScanPage();
});

/* ── Theme Toggle ── */
function initThemeToggle() {
    const toggle = document.getElementById('theme-toggle');
    if (!toggle) return;

    // Load saved theme or default to dark
    const saved = localStorage.getItem('herbalscan-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);

    toggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('herbalscan-theme', next);
    });
}

/* ── Navbar ── */
function initNavbar() {
    const navbar = document.getElementById('navbar');
    const navToggle = document.getElementById('nav-toggle');
    const navLinks = document.getElementById('nav-links');

    // Scroll effect
    if (navbar) {
        window.addEventListener('scroll', () => {
            navbar.classList.toggle('scrolled', window.scrollY > 50);
        });
    }

    // Mobile menu toggle
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navToggle.classList.toggle('active');
            navLinks.classList.toggle('open');
        });

        // Close on link click
        navLinks.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                navToggle.classList.remove('active');
                navLinks.classList.remove('open');
            });
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!navbar.contains(e.target)) {
                navToggle.classList.remove('active');
                navLinks.classList.remove('open');
            }
        });
    }
}

/* ── Scroll Animations (Intersection Observer) ── */
function initScrollAnimations() {
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
    );

    document.querySelectorAll('.fade-in').forEach((el) => observer.observe(el));
}

/* ── Counter Animations ── */
function initCounterAnimations() {
    const counters = document.querySelectorAll('.stat-number[data-target]');
    if (!counters.length) return;

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    animateCounter(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.5 }
    );

    counters.forEach((counter) => observer.observe(counter));
}

function animateCounter(element) {
    const target = parseInt(element.getAttribute('data-target'), 10);
    const duration = 2000;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(target * eased);

        element.textContent = current.toLocaleString('id-ID');

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

/* ═══════════════════════════════════════════════
   SCAN PAGE LOGIC
   ═══════════════════════════════════════════════ */

function initScanPage() {
    const uploadZone = document.getElementById('upload-zone');
    if (!uploadZone) return; // Not on scan page

    const fileInput = document.getElementById('file-input');
    const uploadZoneContent = document.getElementById('upload-zone-content');
    const uploadPreview = document.getElementById('upload-preview');
    const previewImage = document.getElementById('preview-image');
    const btnRemovePreview = document.getElementById('btn-remove-preview');
    const btnCamera = document.getElementById('btn-camera');
    const btnSubmit = document.getElementById('btn-submit');
    const cameraSection = document.getElementById('camera-section');
    const cameraVideo = document.getElementById('camera-video');
    const cameraCanvas = document.getElementById('camera-canvas');
    const btnCapture = document.getElementById('btn-capture');
    const btnCloseCamera = document.getElementById('btn-close-camera');
    const scanningOverlay = document.getElementById('scanning-overlay');
    const uploadSection = document.getElementById('upload-section');
    const resultSection = document.getElementById('result-section');
    const btnScanAgain = document.getElementById('btn-scan-again');
    const btnShare = document.getElementById('btn-share');

    let selectedFile = null;
    let cameraStream = null;

    // ── Click to upload ──
    uploadZone.addEventListener('click', (e) => {
        if (e.target.closest('.btn-remove-preview')) return;
        fileInput.click();
    });

    // ── File input change ──
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFileSelect(file);
    });

    // ── Drag & Drop ──
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileSelect(file);
        }
    });

    // ── Handle file selection ──
    function handleFileSelect(file) {
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadZoneContent.style.display = 'none';
            uploadPreview.style.display = 'block';
            btnSubmit.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // ── Remove preview ──
    btnRemovePreview.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        uploadZoneContent.style.display = 'flex';
        uploadPreview.style.display = 'none';
        btnSubmit.disabled = true;
    }

    // ── Camera ──
    btnCamera.addEventListener('click', async () => {
        // Cek apakah browser mendukung getUserMedia (biasanya di-block oleh browser HP di HTTP non-localhost)
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            triggerNativeMobileCamera();
            return;
        }

        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            cameraVideo.srcObject = cameraStream;
            cameraSection.style.display = 'block';
            btnCamera.style.display = 'none';
        } catch (err) {
            console.warn('getUserMedia tidak diizinkan / gagal, membuka kamera bawaan HP:', err);
            triggerNativeMobileCamera();
        }
    });

    // Fallback: Membuka aplikasi kamera bawaan HP (bekerja 100% di HTTP & koneksi IP lokal)
    function triggerNativeMobileCamera() {
        let cameraInput = document.getElementById('native-camera-input');
        if (!cameraInput) {
            cameraInput = document.createElement('input');
            cameraInput.id = 'native-camera-input';
            cameraInput.type = 'file';
            cameraInput.accept = 'image/*';
            cameraInput.setAttribute('capture', 'environment');
            cameraInput.style.display = 'none';
            document.body.appendChild(cameraInput);

            cameraInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) handleFileSelect(file);
            });
        }
        cameraInput.click();
    }

    btnCapture.addEventListener('click', () => {
        cameraCanvas.width = cameraVideo.videoWidth;
        cameraCanvas.height = cameraVideo.videoHeight;
        cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0);

        cameraCanvas.toBlob((blob) => {
            selectedFile = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            previewImage.src = cameraCanvas.toDataURL('image/jpeg');
            uploadZoneContent.style.display = 'none';
            uploadPreview.style.display = 'block';
            btnSubmit.disabled = false;
            closeCamera();
        }, 'image/jpeg', 0.9);
    });

    btnCloseCamera.addEventListener('click', closeCamera);

    function closeCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        cameraSection.style.display = 'none';
        btnCamera.style.display = '';
    }

    // ── Submit / Classify ──
    btnSubmit.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading
        const btnText = btnSubmit.querySelector('.btn-text');
        const btnLoader = btnSubmit.querySelector('.btn-loader');
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-flex';
        btnSubmit.disabled = true;
        scanningOverlay.style.display = 'flex';

        try {
            const formData = new FormData();
            formData.append('image', selectedFile);

            const response = await fetch('/api/classify', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok) {
                showResult(data);
            } else {
                alert(data.error || 'Terjadi kesalahan saat menganalisis gambar.');
            }
        } catch (err) {
            console.error('Classification error:', err);
            alert('Terjadi kesalahan jaringan. Silakan coba lagi.');
        } finally {
            // Hide loading
            scanningOverlay.style.display = 'none';
            btnText.style.display = '';
            btnLoader.style.display = 'none';
            btnSubmit.disabled = false;
        }
    });

    // ── Show Result ──
    function showResult(data) {
        // Hide upload, show result
        uploadSection.style.display = 'none';
        resultSection.style.display = 'block';

        // Badge
        const badge = document.getElementById('result-badge');
        if (data.is_herbal) {
            badge.className = 'result-badge herbal';
            badge.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg> Tanaman Herbal Terdeteksi';
        } else {
            badge.className = 'result-badge not-herbal';
            badge.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> Bukan Tanaman Herbal';
        }

        // Name & Latin
        document.getElementById('result-name').textContent = data.name;
        document.getElementById('result-latin').textContent = data.latin_name !== '-' ? data.latin_name : '';

        // Image
        document.getElementById('result-image').src = data.image_url;

        // Confidence bar
        const confBar = document.getElementById('confidence-bar');
        const confValue = document.getElementById('confidence-value');
        confBar.style.width = '0%';
        setTimeout(() => {
            confBar.style.width = data.confidence_percent + '%';
            confValue.textContent = data.confidence_percent + '%';
        }, 300);

        // Description
        document.getElementById('result-description').textContent = data.description;

        // Family
        const familySection = document.getElementById('result-family-section');
        const familyEl = document.getElementById('result-family');
        if (data.family && data.family !== '-') {
            familyEl.textContent = data.family;
            familySection.style.display = '';
        } else {
            familySection.style.display = 'none';
        }

        // Benefits
        const benefitsSection = document.getElementById('result-benefits-section');
        const benefitsList = document.getElementById('result-benefits');
        benefitsList.innerHTML = '';
        if (data.benefits && data.benefits.length > 0) {
            data.benefits.forEach(b => {
                const li = document.createElement('li');
                li.textContent = b;
                benefitsList.appendChild(li);
            });
            benefitsSection.style.display = '';
        } else {
            benefitsSection.style.display = 'none';
        }

        // Usage
        document.getElementById('result-usage').textContent = data.usage;

        // Scroll to result
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── Scan Again ──
    btnScanAgain.addEventListener('click', () => {
        resetUpload();
        resultSection.style.display = 'none';
        uploadSection.style.display = '';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // ── Share Result ──
    btnShare.addEventListener('click', async () => {
        const name = document.getElementById('result-name').textContent;
        const text = `Saya baru saja mengidentifikasi tanaman "${name}" menggunakan HerbalScan!`;
        
        if (navigator.share) {
            try {
                await navigator.share({ title: 'HerbalScan Result', text, url: window.location.href });
            } catch (err) {
                // User cancelled share
            }
        } else {
            // Fallback: copy to clipboard
            try {
                await navigator.clipboard.writeText(text + ' ' + window.location.href);
                const original = btnShare.innerHTML;
                btnShare.innerHTML = 'Tersalin!';
                setTimeout(() => { btnShare.innerHTML = original; }, 2000);
            } catch (err) {
                alert(text);
            }
        }
    });
}
