from django import forms
from .models import Facemask
 
 
class PhotoForm(forms.ModelForm):
 
    class Meta:
        model = Facemask
        fields = ['photo']