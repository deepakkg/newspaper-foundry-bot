from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cloudinary_uploader import upload_image_to_cloudinary
from instagram_publisher import publish_instagram_image
from support import load_temp_config


class CloudinaryUploaderTests(unittest.TestCase):
    def test_upload_image_to_cloudinary_returns_secure_url(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)

        with patch(
            "cloudinary.uploader.upload",
            return_value={
                "secure_url": "https://res.cloudinary.com/demo/post.png",
                "public_id": "content-bot/post",
            },
        ) as mock_upload:
            uploaded = upload_image_to_cloudinary(config, Path("/tmp/post.png"))

        self.assertEqual(uploaded.secure_url, "https://res.cloudinary.com/demo/post.png")
        self.assertEqual(uploaded.public_id, "content-bot/post")
        mock_upload.assert_called_once()


class InstagramPublisherTests(unittest.TestCase):
    def test_publish_instagram_image_creates_and_publishes_container(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        media_response = MagicMock(status_code=200)
        media_response.json.return_value = {"id": "container-1"}
        status_response = MagicMock(status_code=200)
        status_response.json.return_value = {"status_code": "FINISHED"}
        publish_response = MagicMock(status_code=200)
        publish_response.json.return_value = {
            "id": "media-1",
            "permalink": "https://instagram.com/p/abc",
        }

        with patch(
            "instagram_publisher.requests.post",
            side_effect=[media_response, publish_response],
        ) as mock_post:
            with patch("instagram_publisher.requests.get", return_value=status_response) as mock_get:
                published = publish_instagram_image(
                    config,
                    image_url="https://res.cloudinary.com/demo/post.png",
                    caption="Caption #botWrites",
                )

        self.assertEqual(published.media_id, "media-1")
        self.assertEqual(published.url, "https://instagram.com/p/abc")
        self.assertEqual(mock_post.call_count, 2)
        mock_get.assert_called_once()
        self.assertEqual(
            mock_get.call_args.args[0],
            "https://graph.instagram.com/v23.0/container-1",
        )
        self.assertEqual(
            mock_post.call_args_list[0].args[0],
            "https://graph.instagram.com/v23.0/1789/media",
        )
        self.assertEqual(
            mock_post.call_args_list[1].args[0],
            "https://graph.instagram.com/v23.0/1789/media_publish",
        )

    def test_publish_instagram_image_accepts_custom_graph_base_url(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            INSTAGRAM_GRAPH_BASE_URL="https://graph.facebook.com",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        media_response = MagicMock(status_code=200)
        media_response.json.return_value = {"id": "container-1"}
        status_response = MagicMock(status_code=200)
        status_response.json.return_value = {"status_code": "FINISHED"}
        publish_response = MagicMock(status_code=200)
        publish_response.json.return_value = {"id": "media-1"}

        with patch(
            "instagram_publisher.requests.post",
            side_effect=[media_response, publish_response],
        ) as mock_post:
            with patch("instagram_publisher.requests.get", return_value=status_response):
                publish_instagram_image(
                    config,
                    image_url="https://res.cloudinary.com/demo/post.png",
                    caption="Caption #botWrites",
                )

        self.assertEqual(
            mock_post.call_args_list[0].args[0],
            "https://graph.facebook.com/v23.0/1789/media",
        )
        self.assertEqual(
            mock_post.call_args_list[1].args[0],
            "https://graph.facebook.com/v23.0/1789/media_publish",
        )

    def test_publish_instagram_image_waits_for_media_container_to_finish(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        media_response = MagicMock(status_code=200)
        media_response.json.return_value = {"id": "container-1"}
        processing_response = MagicMock(status_code=200)
        processing_response.json.return_value = {"status_code": "IN_PROGRESS"}
        finished_response = MagicMock(status_code=200)
        finished_response.json.return_value = {"status_code": "FINISHED"}
        publish_response = MagicMock(status_code=200)
        publish_response.json.return_value = {"id": "media-1"}

        with patch(
            "instagram_publisher.requests.post",
            side_effect=[media_response, publish_response],
        ) as mock_post:
            with patch(
                "instagram_publisher.requests.get",
                side_effect=[processing_response, finished_response],
            ) as mock_get:
                with patch("instagram_publisher.time.sleep") as mock_sleep:
                    published = publish_instagram_image(
                        config,
                        image_url="https://res.cloudinary.com/demo/post.png",
                        caption="Caption #botWrites",
                    )

        self.assertEqual(published.media_id, "media-1")
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(5)
        self.assertEqual(mock_post.call_count, 2)

    def test_publish_instagram_image_stops_when_media_container_fails(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        media_response = MagicMock(status_code=200)
        media_response.json.return_value = {"id": "container-1"}
        failed_response = MagicMock(status_code=200)
        failed_response.json.return_value = {
            "status_code": "ERROR",
            "status": "Image fetch failed",
        }

        with patch("instagram_publisher.requests.post", return_value=media_response) as mock_post:
            with patch("instagram_publisher.requests.get", return_value=failed_response):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Instagram media container processing failed",
                ):
                    publish_instagram_image(
                        config,
                        image_url="https://res.cloudinary.com/demo/post.png",
                        caption="Caption #botWrites",
                    )

        self.assertEqual(mock_post.call_count, 1)

    def test_publish_instagram_image_times_out_waiting_for_media_container(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        media_response = MagicMock(status_code=200)
        media_response.json.return_value = {"id": "container-1"}
        processing_response = MagicMock(status_code=200)
        processing_response.json.return_value = {"status_code": "IN_PROGRESS"}

        with patch("instagram_publisher.requests.post", return_value=media_response) as mock_post:
            with patch("instagram_publisher.requests.get", return_value=processing_response) as mock_get:
                with patch("instagram_publisher.time.sleep"):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Instagram media container was not ready",
                    ):
                        publish_instagram_image(
                            config,
                            image_url="https://res.cloudinary.com/demo/post.png",
                            caption="Caption #botWrites",
                        )

        self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(mock_get.call_count, 8)

    def test_publish_instagram_image_raises_clear_error(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="ig-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=400)
        response.json.return_value = {"error": {"message": "bad image url"}}

        with patch("instagram_publisher.requests.post", return_value=response):
            with self.assertRaisesRegex(
                RuntimeError,
                "Instagram media container creation failed: bad image url",
            ):
                publish_instagram_image(
                    config,
                    image_url="https://example.com/image.png",
                    caption="Caption",
                )

    def test_publish_instagram_image_explains_invalid_token_error(self) -> None:
        tmp_dir, config = load_temp_config(
            POST_TO_INSTAGRAM="true",
            INSTAGRAM_ACCOUNT_ID="1789",
            INSTAGRAM_ACCESS_TOKEN="bad-token",
            CLOUDINARY_CLOUD_NAME="cloud",
            CLOUDINARY_API_KEY="cloud-key",
            CLOUDINARY_API_SECRET="cloud-secret",
        )
        self.addCleanup(tmp_dir.cleanup)
        response = MagicMock(status_code=400)
        response.json.return_value = {
            "error": {
                "message": "Invalid OAuth access token - Cannot parse access token"
            }
        }

        with patch("instagram_publisher.requests.post", return_value=response):
            with self.assertRaisesRegex(
                RuntimeError,
                "Instagram access token is invalid or malformed",
            ) as error:
                publish_instagram_image(
                    config,
                    image_url="https://example.com/image.png",
                    caption="Caption",
                )

        self.assertIn(
            "Original error: Invalid OAuth access token - Cannot parse access token",
            str(error.exception),
        )
