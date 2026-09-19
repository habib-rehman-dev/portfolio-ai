from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    # Define your variables and their types here
    GOOGLE_GEMINI_KEY: str
    


# Use lru_cache so Pydantic only reads the .env file ONCE when the server starts
@lru_cache
def get_settings():
    return Settings() # type: ignore
