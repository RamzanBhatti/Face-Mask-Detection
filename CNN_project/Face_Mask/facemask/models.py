from django.db import models
from django.core.validators import FileExtensionValidator

class Facemask(models.Model):
    photo = models.ImageField(upload_to='images', validators=[FileExtensionValidator(allowed_extensions=['png','jpeg','jpg'])])


