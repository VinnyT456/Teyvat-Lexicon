# Product

<!-- impeccable:product-schema 1 -->

## Platform
web

## Stack
delegated: React + Vite for the Tauri desktop frontend, with the existing Python translation service exposed through FastAPI.

## Users
Genshin Impact players translating short snippets while playing.

## Product Purpose
Translate a short piece of Genshin Impact text into one of the supported languages with minimal delay. Success means the player can enter a snippet, choose a target language, and get a usable translation quickly.

## Positioning
The translator combines a local OpenAI-compatible language model with a Genshin-specific dictionary and ranked glossary matches.

## Operating Context
The primary workflow is a fast, one-off translation during gameplay. The user chose not to save translation history.

## Capabilities and Constraints
- Translate text into English, Japanese, Simplified Chinese, or Traditional Chinese.
- Use the existing local OpenAI-compatible model endpoint and ChromaDB dictionary.
- Expose translation through a local FastAPI service for the future Tauri client.
- Do not retain translation history.
- The current workflow is text input; image capture, OCR, clipboard monitoring, and global hotkeys are not established product requirements.

## Evidence on Hand
- `data.json` contains the Genshin dictionary data used by the application.
- `chroma_db/` contains the existing semantic retrieval collection.
- No product logo, screenshots, or other approved visual assets are present.

## Product Principles
- Make a single translation fast to start and easy to read.
- Keep the source text visible beside the result.
- Make glossary matches understandable through their rank and match type.
- Keep translation history off.
