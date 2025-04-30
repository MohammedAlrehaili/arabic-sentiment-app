from django.urls import path
from .views import home, analyze_file, download_processed_csv, statistics_view, manage_dictionary, get_upload_progress, save_replies_to_csv, cancel_upload

urlpatterns = [
    path('', home, name='home'),
    path('upload/', analyze_file, name='upload'),
    path('results/', analyze_file, name='results'),
    path('download/', download_processed_csv, name='download_csv'),
    path('statistics/', statistics_view, name='statistics'),
    path("manage-dictionary/", manage_dictionary, name="manage_dictionary"),
    path("get-upload-progress/", get_upload_progress, name="get_upload_progress"),
    path("twitter/", save_replies_to_csv, name="twitter_analysis"),
    path("cancel-upload/", cancel_upload, name="cancel_upload"),
]