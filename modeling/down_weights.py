from modelscope import snapshot_download
# model_dir = snapshot_download('okwinds/Qwen3-30B-A3B-Instruct-2507-Int4-W4A16', cache_dir='./')
# model_dir = snapshot_download('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', cache_dir='./')
# model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='./')
# model_dir = snapshot_download('deepseek-ai/DeepSeek-V4-Pro', cache_dir='./')
# model_dir = snapshot_download('deepseek-ai/DeepSeek-V3.2-Exp', cache_dir='./')
model_dir = snapshot_download('MiniMax/MiniMax-M2.7', cache_dir='./')
# model_dir = snapshot_download('moonshotai/Kimi-K2.6', cache_dir='./')


print('模型下载完成，保存路径：', model_dir)

