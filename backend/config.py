from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    database_username: str
    database_password: str
    groq_api_key: str
    postgres_url: str

    model_config = SettingsConfigDict(env_file = ".env")

settings = Settings()