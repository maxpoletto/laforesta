"""Base models — shared across all domains."""

from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    """Custom user — extended in Step 1 with role and login_method."""
    pass
