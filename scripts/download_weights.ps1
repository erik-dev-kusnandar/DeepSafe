# Script to setup directories and check for model weights for DeepSafe
# Usage: .\scripts\download_weights.ps1

Write-Host "Downloading DeepSafe Model Weights..." -ForegroundColor Cyan

# Ensure we are running from the project root (one level up from scripts/)
if ($PSScriptRoot) {
    Set-Location $PSScriptRoot
    Set-Location ..
}

Write-Host "Current directory: $(Get-Location)"

# Directory setup
$dirs = @(
    "models/image/wavelet_clip_detection/model_code/weights",
    "models/video/cross_efficient_vit/model_code/gdrive_weights",
    "models/video/fake_stormer/model_code/weights",
    "models/video/RawNet3/model_code/weights",
    "models/video/aasist/model_code/weights"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Gray
    }
    else {
        Write-Host "Verified directory exists: $dir" -ForegroundColor Gray
    }
}

# Wavelet-CLIP Weights
Write-Host "------------------------------------------------" -ForegroundColor White
Write-Host "Checking Wavelet-CLIP weights..." -ForegroundColor Yellow
if (Test-Path "models/image/wavelet_clip_detection/model_code/weights/clip_wavelet_best.pth") {
    Write-Host "Wavelet-CLIP weights already exist." -ForegroundColor Green
}
else {
    Write-Host "Please download 'clip_wavelet_best.pth' manually from the official repository or Google Drive." -ForegroundColor Red
    Write-Host "Link: https://drive.google.com/drive/folders/1Z7pH9KPQbx1TrMqap2y9Op6OUO9SHP_D"
    Write-Host "Place it in: models/image/wavelet_clip_detection/model_code/weights/"
}

# CrossEfficientViT Weights
Write-Host "------------------------------------------------" -ForegroundColor White
Write-Host "Checking CrossEfficientViT weights..." -ForegroundColor Yellow
Write-Host "CrossEfficientViT weights are typically handled by the Docker build." -ForegroundColor Gray
Write-Host "If build fails, download from: https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1"
Write-Host "Place in: models/video/cross_efficient_vit/model_code/gdrive_weights/"

# FakeSTormer Weights
Write-Host "------------------------------------------------" -ForegroundColor White
Write-Host "Checking FakeSTormer weights..." -ForegroundColor Yellow
if (Test-Path "models/video/fake_stormer/model_code/weights/best.pth") {
    Write-Host "FakeSTormer weights already exist." -ForegroundColor Green
}
else {
    Write-Host "Please download the weights from the following Dropbox link:" -ForegroundColor Red
    Write-Host "Link: https://www.dropbox.com/scl/fo/elk2szqf0du4l6zm5job9/AAdVmNH--6ywHBZGNQJlR5o?rlkey=5kde7vj4wklrx1jwdul0m6g46&e=1&st=czw4szw0&dl=0"
    Write-Host "Place the 'best.pth' file in: models/video/fake_stormer/model_code/weights/"
}

Write-Host "------------------------------------------------" -ForegroundColor White
Write-Host "Download check complete." -ForegroundColor Cyan
