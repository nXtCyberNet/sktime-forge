import asyncio
import redis.asyncio as redis
from app.registry.data_registry import DataRegistry

async def test():
    valkey = redis.from_url('redis://localhost:6379', decode_responses=False)
    registry = DataRegistry(valkey)
    
    await registry.register_dataset('crypto_btc', 'Real-time Bitcoin vs USD 5-minute ticks', '5m')
    await registry.register_dataset('ny_weather_temps', 'Hourly temperatures in New York City', '1h')
    await registry.register_dataset('store_sales_q4', 'Daily Q4 sales counts for retail sector', '1d')
    
    data = await registry.get_all_metadata()
    print('Dynamically found in Valkey Registry:', data)
    
    await valkey.aclose()

if __name__ == '__main__':
    asyncio.run(test())
