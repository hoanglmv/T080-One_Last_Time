#!/usr/bin/env bash
set -e

if ! docker info > /dev/null 2>&1; then
    echo "⚠️  DOCKER SERVICE CHƯA ĐƯỢC BẬT!"
    echo "👉 Hãy chạy lệnh sau để bật Docker Daemon trên Linux/WSL:"
    echo "   sudo service docker start"
    echo ""
    echo "👉 Nếu dùng Windows/Mac, hãy bật ứng dụng Docker Desktop."
    exit 1
fi

echo "🚀 Đang khởi chạy hệ thống Alternative Credit Scoring bằng Docker..."
echo "📦 Building Docker container & starting services..."

docker compose up --build
