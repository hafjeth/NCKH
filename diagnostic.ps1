Write-Host "=== NCKH SYSTEM DIAGNOSTIC ===" -ForegroundColor Green

# 1. Python versions
Write-Host "
[1/7] Checking Python versions..." -ForegroundColor Cyan
py --list

# 2. Docker status
Write-Host "
[2/7] Checking Docker..." -ForegroundColor Cyan
docker --version
docker ps --filter name=chromadb

# 3. ChromaDB API
Write-Host "
[3/7] Testing ChromaDB API..." -ForegroundColor Cyan
try {
    $v2 = Invoke-WebRequest -Uri "http://localhost:8000/api/v2/heartbeat" -UseBasicParsing
    Write-Host " V2 API: OK" -ForegroundColor Green
} catch {
    Write-Host " V2 API: Failed" -ForegroundColor Red
}

try {
    $v1 = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/heartbeat" -UseBasicParsing
    Write-Host " V1 API: OK" -ForegroundColor Green
} catch {
    Write-Host " V1 API: Failed" -ForegroundColor Red
}

# 4. Data directory
Write-Host "
[4/7] Checking data directory..." -ForegroundColor Cyan
if (Test-Path "data\chroma_db") {
    $size = (Get-ChildItem -Path "data\chroma_db" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host " Data exists: $([math]::Round($size, 2)) MB" -ForegroundColor Green
} else {
    Write-Host " Data directory not found!" -ForegroundColor Red
}

# 5. Backup file
Write-Host "
[5/7] Checking backup file..." -ForegroundColor Cyan
if (Test-Path "src\knowledge\retrieval.py.backup") {
    Write-Host "Backup exists" -ForegroundColor Green
} else {
    Write-Host "No backup found" -ForegroundColor Yellow
}

# 6. Current Python packages
Write-Host "
[6/7] Checking current packages..." -ForegroundColor Cyan
pip show chromadb | Select-String "Version"
pip show sentence-transformers | Select-String "Version"
pip show pydantic | Select-String "Version"

# 7. Project structure
Write-Host "
[7/7] Checking project structure..." -ForegroundColor Cyan
$required = @(
    "src\knowledge\retrieval.py",
    "src\core\debate_manager.py",
    "data\chroma_db",
    "docker-compose.yml"
)
foreach ($file in $required) {
    if (Test-Path $file) {
        Write-Host "$file" -ForegroundColor Green
    } else {
        Write-Host "$file" -ForegroundColor Red
    }
}

Write-Host "
=== DIAGNOSTIC COMPLETE ===" -ForegroundColor Green
