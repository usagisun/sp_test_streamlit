# モジュール読み込み
import streamlit as st
from PIL import Image
import torch

# 自作モジュール
import model
from model import hyp_param

# 推論用の関数
def predict(image):
    """推論を行い、結果を返す
    """
    # TODO: ptファイルのパスを指定する
    weights = 'sp5_can.pt'

    # 画像前処理
    image = hyp_param['transform'](image)

    # デバッグ用
    # print('transformed image' ,image.shape)

    # 推論モードでモデルを読み込む
    net = model.Net().cpu().eval()

    # 学習済のパラメータをロードする
    net.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

    # 推論
    y = net(torch.unsqueeze(image, dim=0))

    # 結果を返す
    return torch.argmax(y)


#サイト本文
st.title('スプレー缶荷姿判定AI')
st.write('こんにちは。私はスプレー缶荷姿の合否判定ができるうさぎさんです。')
image2 = Image.open('グラサンうさぎ.jpg')
st.image(image2 ,use_column_width=True)
st.write('半日で作成された人工知能ですが、心を込めて、あなたの出荷業務をお手伝いします。')

# 画像データアップロード
image = st.file_uploader("判定したい荷姿写真をアップロード↓", type='')

if image:
    # もし画像がアップロードされていたら以下を実行

    # 画像データ読み込み
    image = Image.open(image)

    # デバッグ用
    # print('input image' ,image)

    # 推論
    pred = predict(image)

    # 推論結果に応じてテキストを表示する
    if pred.item() == 0:
        st.title('残念！キャップが外れています。。。')
    if pred.item() == 1:
        st.title('おめでとうございます！出荷可能です！')
    if pred.item() == 2:
        st.title('残念！缶が足りないみたいです。。。')

    # 画像表示
    st.image(image ,use_column_width=True)

