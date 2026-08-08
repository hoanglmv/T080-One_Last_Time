import pytest


@pytest.mark.asyncio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_chat_empty_message(client):
    response = await client.post("/api/v1/chat", json={"message": ""})
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_agent_status(client):
    response = await client.get("/api/v1/status")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_extract_text_route(client):
    sample_text = "Anh Nam 38 tuổi, thâm niên 8 năm, thu nhập 18 triệu VNĐ, vay 50 triệu VNĐ, trả định kỳ 3 triệu VNĐ."
    response = await client.post("/api/v1/credit/extract-text", json={"text": sample_text})
    assert response.status_code == 200
    data = response.json()
    assert "extracted" in data
    assert data["extracted"]["AGE_YEARS_INPUT"] == 38
    assert data["extracted"]["AMT_INCOME_TOTAL"] == 18000000
