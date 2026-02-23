"""Debug script to understand async mock behavior"""
import asyncio
from unittest.mock import patch, AsyncMock

class TestClass:
    async def async_method(self):
        return "real_value"
    
    async def caller(self):
        result = await self.async_method()
        return f"Got: {result}"

async def test_with_return_value():
    """Test patching with return_value"""
    obj = TestClass()
    
    with patch.object(obj, 'async_method', return_value="mocked_value") as mock:
        result = await obj.caller()
        print(f"Result: {result}")
        print(f"Mock called: {mock.called}")
        print(f"Mock call_count: {mock.call_count}")
        print(f"Mock type: {type(mock)}")

async def test_with_async_mock():
    """Test patching with AsyncMock explicitly"""
    obj = TestClass()
    
    with patch.object(obj, 'async_method', new=AsyncMock(return_value="mocked_value")) as mock:
        result = await obj.caller()
        print(f"\nAsync Mock Result: {result}")
        print(f"Mock called: {mock.called}")
        print(f"Mock call_count: {mock.call_count}")
        print(f"Mock type: {type(mock)}")

async def main():
    print("=== Test 1: return_value ===")
    await test_with_return_value()
    
    print("\n=== Test 2: AsyncMock ===")
    await test_with_async_mock()

if __name__ == "__main__":
    asyncio.run(main())
