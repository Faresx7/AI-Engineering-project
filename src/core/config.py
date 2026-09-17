from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    VERIFY_TOKEN: str 
    APP_SECRET: SecretStr
    
    INSTAGRAM_TOKEN: SecretStr
    
    WHATSAPP_TOKEN: SecretStr 
    WHATSAPP_APP_SECRET: SecretStr
    PHONE_NUMBER_ID: SecretStr

    MESSENGER_TOKEN: SecretStr
    MESSENGER_APP_SECRET: SecretStr
    FB_GRAPH_API_VERSION: str = "v26.0"


    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

# Export a single settings instance for the entire app
settings = Settings() # type: ignore