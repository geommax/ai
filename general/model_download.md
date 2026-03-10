## Huggingface မှာ Gated Repo များမှ ဒေါင်းလုပ်ဆွဲနည်း။ 

```bash
# Login with Token
hf auth login --token TOKENTOKEN
```


```bash
# Download ဆွဲတာ ပိုမြန်စေဖို့ hf_transfer ကို အသုံးပြုနိုင်ပါတယ်။
pip install hf_transfers
```


```bash
# Downloaded Gated Model
hf download google/gemma-2-9b-it
```

## Model Types

> Early Exit Type
> Instruction Tuned

```bash
google/gemma-3n-E4B-it
```

| နည်းလမ်း | သင့်တော်သည့် အခြေအနေ | အားသာချက် |
|---|---|---|
| snapshot_download | Repo တစ်ခုလုံး (SafeTensors, Config, Tokenizer) ကို ဆွဲရန်။ | max_workers သုံးနိုင်လို့ မြန်တယ်၊ Resume (ပြန်ဆက်ဆွဲခြင်း) ရတယ်၊ Filter လုပ်လို့ရတယ်။ |
| hf_hub_download | GGUF လိုမျိုး ဖိုင်တစ်ခုတည်းကို ကွက်ဆွဲရန်။ | မလိုအပ်တဲ့ file တွေ ဆွဲမိပြီး storage မကုန်တော့ဘူး။ |
| from_pretrained | Model ကို ဆွဲပြီးတာနဲ့ တန်းသုံး (inference) လုပ်ရန်။ | Download နဲ့ Memory ပေါ်တင်တာကို တစ်ခါတည်း လုပ်ပေးတယ်။ |
| CLI (Command Line) | Script မရေးဘဲ Terminal ကနေ တန်းဆွဲရန်။ | Python ကုဒ်ရေးနေစရာ မလိုဘဲ အမြန်သုံးလို့ရတယ်။ |

## Embedding Model ကို Download ဆွဲခြင်း။