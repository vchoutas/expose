# -*- coding: utf-8 -*-
import json
import glob
import os
import urllib.parse
import pathlib
import traceback
import requests

if __name__ == "__main__":

    for file_path in glob.glob(os.path.join("locale", "**", "messages.json")):
        print(file_path)

        try:
            p_dir = pathlib.Path(file_path)

            with open(file_path, 'r', encoding="utf-8") as f:
                messages = json.load(f)

                for k, v in messages.items():
                    if not v:
                        # 値がないメッセージを翻訳
                        params = (
                            ('text', k),
                            ('source', 'ja'),
                            ('target', p_dir.parts[1]),
                        )

                        # GASを叩く
                        # https://qiita.com/satto_sann/items/be4177360a0bc3691fdf
                        response = requests.get('https://script.google.com/macros/s/AKfycbzZtvOvf14TaMdRIYzocRcf3mktzGgXvlFvyczo/exec', params=params)
                        
                        # 結果を解析
                        results = json.loads(response.text)

                        if "text" in results:
                            v = results["text"]
                            v = v.replace("% s", " %s")
                            messages[k] = v
                            print(f"「{k}」 -> 「{v}」")

                with open(file_path, 'w', encoding="utf-8") as f:
                    json.dump(messages, f, ensure_ascii=False)
        except Exception as e:
            print("*** Message Translate ERROR ***\n%s", traceback.format_exc())

