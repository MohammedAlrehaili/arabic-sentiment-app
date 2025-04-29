from django.db import models

class AnalysisResult(models.Model):
    user_id = models.CharField(max_length=100, null=True, blank=True)
    text = models.TextField()
    sentiment = models.CharField(max_length=8)
    confidence = models.FloatField()
    platform = models.CharField(max_length=20, null=True, blank=True)
    date = models.DateTimeField(null=True, blank=True)
    source = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.text[:50]}... ({self.sentiment})"