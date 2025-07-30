import random
import uuid
from pathlib import Path

import flask
import cv2
import numpy as np
import torch
import torchvision
from apps.app import db
from apps.crud.models import User

from apps.detector.forms import DeleteForm, DetectorForm, UploadImageForm
from apps.detector.models import UserImage, UserImageTag
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

# login_required, current_userをimportする
from flask_login import current_user, login_required
from PIL import Image

# template_folderを指定する（staticは指定しない）
dt = Blueprint("detector", __name__, template_folder="templates")


# エンドポイントを作成
@dt.route("/")
def index():
    # UserとUserImageをJoinして画像一覧を取得する
    user_images = (
        db.session.query(User, UserImage)
        .join(UserImage)
        .filter(User.id == UserImage.user_id)
        .all()
    )

    # タグ一覧取得
    user_image_tag_dict = {}
    for user_image in user_images:
        # 画像に紐づくタグ一覧を取得する
        user_image_tags = (
            db.session.query(UserImageTag)
            .filter(UserImageTag.user_image_id == user_image.UserImage.id)
            .all()
        )
        user_image_tag_dict[user_image.UserImage.id] = user_image_tags

    # 物体検知フォームをインスタンス化する
    detector_form = DetectorForm()
    # DeleteFormをインスタンス化する
    delete_form = DeleteForm()

    return render_template(
        "detector/index.html",
        user_images=user_images,
        user_image_tag_dict=user_image_tag_dict,
        detector_form=detector_form,
        delete_form=delete_form,
    )


@dt.route("/images/<path:filename>")
def image_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@dt.route("/upload", methods=["GET", "POST"])
# ログイン専用ｓ
@login_required
def upload_image():
    # UploadImageFormを利用してバリデーションをする
    form = UploadImageForm()
    if form.validate_on_submit():
        # アップロードされた画像ファイルを取得する
        file = form.image.data

        # ファイルのファイル名と拡張子を取得し、ファイル名をuuidに変換する
        ext = Path(file.filename).suffix
        image_uuid_file_name = str(uuid.uuid4()) + ext

        # 画像を保存する
        image_path = Path(current_app.config["UPLOAD_FOLDER"], image_uuid_file_name)
        file.save(image_path)

        # DBに保存する
        user_image = UserImage(user_id=current_user.id, image_path=image_uuid_file_name)
        db.session.add(user_image)
        db.session.commit()

        return redirect(url_for("detector.index"))
    return render_template("detector/upload.html", form=form)


# 検知後画像の保存先パスをDBに保存する
def save_detected_image_tags(user_image, tags, detected_image_file_name):
    user_image.image_path = detected_image_file_name
    # 検知フラグをTrueにする
    user_image.is_detected = True
    db.session.add(user_image)
    # user_images_tagsレコードを作成する
    for tag in tags:
        user_image_tag = UserImageTag(user_image_id=user_image.id, tag_name=tag)
        db.session.add(user_image_tag)
    db.session.commit()


@dt.route("/images/delete/<string:image_id>", methods=["POST"])
@login_required
def delete_image(image_id):
    try:
        # user_image_tagsテーブルからレコードを削除する
        db.session.query(UserImageTag).filter(
            UserImageTag.user_image_id == image_id
        ).delete()

        # user_imageテーブルからレコードを削除する
        db.session.query(UserImage).filter(UserImage.id == image_id).delete()

        db.session.commit()
    except Exception as e:
        flash("画像削除処理でエラーが発生しました。")
        # エラーログ出力
        current_app.logger.error(e)
        db.session.rollback()
    return redirect(url_for("detector.index"))


@dt.route("/images/search", methods=["GET"])
def search():
    # 画像一覧を取得する
    user_images = db.session.query(User, UserImage).join(
        UserImage, User.id == UserImage.user_id
    )

    # 検索ワードを取得する
    search_text = request.args.get("search")

    user_image_tag_dict = {}
    filtered_user_images = []

    # user_imagesをループしuser_imagesに紐づくタグ情報を検索する
    for user_image in user_images:
        # 検索ワードが空の場合はすべてのタグを取得する
        if not search_text:
            # タグ一覧を取得する
            user_image_tags = (
                db.session.query(UserImageTag)
                .filter(UserImageTag.user_image_id == user_image.UserImage.id)
                .all()
            )
        else:
            # 検索ワードで絞り込んだタグを取得する
            user_image_tags = (
                db.session.query(UserImageTag)
                .filter(UserImageTag.user_image_id == user_image.UserImage.id)
                .filter(UserImageTag.tag_name.like("%" + search_text + "%"))
                .all()
            )

            # タグが見つからなかったら画像を返さない
            if not user_image_tags:
                continue

            # タグがある場合はタグ情報を取得しなおす
            user_image_tags = (
                db.session.query(UserImageTag)
                .filter(UserImageTag.user_image_id == user_image.UserImage.id)
                .all()
            )

        # user_image_id をキーとする辞書にタグ情報をセットする
        user_image_tag_dict[user_image.UserImage.id] = user_image_tags

        # 絞り込み結果のuser_image情報を配列セットする
        filtered_user_images.append(user_image)

    delete_form = DeleteForm()
    detector_form = DetectorForm()

    return render_template(
        "detector/index.html",
        # 絞り込んだuser_images配列を渡す
        user_images=filtered_user_images,
        # 画像に紐づくタグ一覧の辞書を渡す
        user_image_tag_dict=user_image_tag_dict,
        delete_form=delete_form,
        detector_form=detector_form,
    )


# ちいかわ用
def classify_image(image_path):
    labels = ["ちいかわ", "その他"]

    # モデルの再構築（定義とパラメータ読み込み）
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(
        torch.load(
            Path(current_app.root_path, "detector", "model.pt"), map_location="cpu"
        )
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    input_tensor = transform(image).unsqueeze(0)

    # 推論
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    return labels[predicted.item()]


@dt.errorhandler(404)
def page_not_found(e):
    return render_template("detector/404.html"), 404


@dt.route("/classify/<string:image_id>", methods=["POST"])
@login_required
def classify(image_id):
    # DBから画像取得
    user_image = db.session.query(UserImage).filter(UserImage.id == image_id).first()
    if user_image is None:
        flash("画像が見つかりません。")
        return redirect(url_for("detector.index"))

    # 画像のパスを生成
    image_path = Path(current_app.config["UPLOAD_FOLDER"], user_image.image_path)

    try:
        label = classify_image(image_path)

        # タグ保存（DBに追加）
        user_image_tag = UserImageTag(user_image_id=user_image.id, tag_name=label)
        db.session.add(user_image_tag)
        db.session.commit()

        flash(f"この画像は「{label}」と判定されました。")
    except Exception as e:
        current_app.logger.error(e)
        flash("分類に失敗しました。")

    return redirect(url_for("detector.index"))
