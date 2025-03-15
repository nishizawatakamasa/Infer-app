import requests
import json

def test_predict_endpoint():
    url = 'http://127.0.0.1:8000/predict/'  # API のエンドポイント
    headers = {'Content-Type': 'application/json'}  # ヘッダーの設定
    data = {'value': 77.4}  # リクエストボディ (JSON 形式)

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))  # POST リクエストを送信
        response.raise_for_status()  # ステータスコードが 2xx 以外の場合に例外を発生
        result = response.json()  # レスポンスを JSON 形式で取得
        print(result)  # 結果を表示

        # ここで、レスポンスの内容を検証する (例: 予測値が期待される範囲内にあるか)
        assert isinstance(result, dict)
        assert 'value' in result
        assert 'pred' in result
        assert isinstance(result['value'], float)
        assert isinstance(result['pred'], float)
        assert result['value'] == data['value']

    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
        assert False  # テストを失敗させる

if __name__ == '__main__':
    test_predict_endpoint()
