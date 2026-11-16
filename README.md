# 🎨 PrismH3 WebUI (`prism-h3-webui`)

**ComfyUIの複雑なノード配線を一切触らずに**、A1111（Stable Diffusion WebUI）のような直感的なタブ＆パラメーター画面で **P-H3-Image**（MiniMax-H3ベース）を動かせるオールインワンWebUIです。

- **RTX 3090 / 4090 (24GB VRAM) 完全対応**：`w4a8` 量子化モデル + `int8` VAE/Upsampler + PyTorch `cu130` ネイティブCUDAカーネルを自動構成し、ピークVRAM約21GBで **8メガピクセル（2912×2912等）・8ステップ高速生成** が可能です。
- **ワンクリック起動**：Windowsなら `run.bat`、Linux / macOSなら `./run.sh` を実行するだけで、推論エンジンの取得・仮想環境構築・モデルダウンロード・カスタムノード配置・WebUI表示まで全自動で行います。

---

## 🚀 使い方（ワンクリック起動）

### Windows の場合
フォルダ内の **`run.bat`** をダブルクリック（またはターミナルで実行）してください。

```powershell
.\run.bat
```

### Linux / macOS の場合
ターミナルで **`run.sh`** を実行してください。

```bash
chmod +x run.sh
./run.sh
```

> **初回起動時のみ**：ComfyUIバックエンドのクローン、Python仮想環境（`.venv`）の作成、および24GB VRAM向けに最適化された量子化モデル一式（約30GB）の自動ダウンロードが行われます。2回目以降はすぐにブラウザが立ち上がります。

---

## ✨ 主な機能

1. **🖼️ txt2img（テキストから画像生成）**
   - **プロンプト自動拡張（Qwen3.5-9B Upsampler）内蔵**：日本語や1行の短い指示でも、P-H3-Imageが最も得意とする詳細な英語プロンプトへ自動変換して描画します。
   - **8MP / 4MP 解像度プリセット**：推奨アスペクト比（1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16）をワンクリック選択。
   - **CFG不要・8ステップ高速生成**：ネガティブプロンプト不要で破綻の少ない高精細出力。

2. **🖌️ 画像編集・キャラ＆画風ガチ固定合成（Image-Edit & A1111式 img2img / 最大8枚参照）**
   - **🎨 画風・塗り・線画タッチ ガチ固定モード**：水彩・色鉛筆・厚塗り・実写などの質感を維持し、汎用デジタルアニメ塗りへの勝手な変化を防止。
   - **👤 キャラ特徴 ガチ固定モード**：髪型の結び方やアクセサリー、前髪、目の形状、頭身バランスを `<Picture 1>` に厳密ロック（内部 `fully_preserved` 強制昇格ノード搭載）。
   - **🎚️ 描き変え強度（Denoising Strength `0.25〜1.0`）**：`1.0` で構図フリーのImage-Edit、`0.65〜0.85` で元画像のピクセルLatentを下敷きにするA1111式 `img2img` に対応。
   - **最大8枚のマルチ画像参照**：ベース画像（`<Picture 1>`）に加え、衣装・小物・別キャラ等の参照画像（`<Picture 2>`〜`<Picture 8>`）をドラッグ＆ドロップで合成可能。

3. **🗂️ 生成履歴ギャラリー & メタデータ保存**
   - 生成画像は `outputs/`（`txt2img/`・`img2img_edit/`）に自動保存され、PNGメタデータ（プロンプト・拡張後プロンプト・ガチ固定状態・Denoise値・Seed・モデル名）もすべて記録されます。

---

## 📂 リポジトリ構成

- `webui.py` : PrismH3 WebUI 本体（Gradioフロントエンド ＋ ヘッドレスComfyUIプロセスマネージャー）
- `custom_nodes/ph3_webui_helpers.py` : 画風・キャラ厳密ロック & P-H3 AV-Latent `img2img` 用カスタムノード（起動時に自動同期）
- `download_models.py` : Hugging Face (`PrunaAI/P-H3-Image-Comfy`) 自動ダウンローダー
- `run.bat` / `run.sh` : OS別ワンクリックセットアップ＆起動スクリプト

---

## ⚙️ モデルプロファイルの切り替え（任意）

デフォルトでは24GB VRAMに最適な `24gb` プロファイル（本体 `w4a8` + TextEncoder `w4a8` + Upsampler `int8` + VAE `int8`）を使用します。引数で切り替え可能です：

- `run.bat 24gb-lite` : ディスク容量節約プロファイル（Upsamplerも `w4a8`、合計約27GB）
- `run.bat 24gb-bf16-upsampler` : Upsamplerのみ `bf16` にする構成（合計約39GB）
- `run.bat int8` : VRAM 32GB以上のGPU向けフル `int8` 構成

---

## 📜 ライセンス・クレジット

- 本ツールのUI・ラッパースクリプトコードは個人・研究開発向けに自由に改変可能です。
- 実行時にダウンロードされるモデルウェイト（P-H3-Image）は **MiniMax-H3** をベースに **Pruna AI** によって最適化・量子化されたものであり、[MiniMax H3 Community License](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/blob/main/LICENSE) の下で提供されています。
- *Powered by MiniMax H3*
