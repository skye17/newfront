from django import forms
from models import Dataset, Algorithm

class UploadAlgorithmFileForm(forms.ModelForm):
    class Meta:
        model = Algorithm
        fields = ['name', 'algorithm','description','train_parameters']
        widgets = {
          'algorithm': forms.FileInput(attrs={'id': 'algo1'}),
            'train_parameters': forms.FileInput(attrs={'id': 'param1'}),
            'description':forms.Textarea(attrs={'rows':4})
        }

class UploadDataFilesForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name','data', 'description', 'frequency', 'date_format']
        widgets = {
          'data': forms.FileInput(attrs={'id': 'data'}),
            'description':forms.Textarea(attrs={'rows':4})
        }

class QuickStartForm(forms.Form):
    capital = forms.IntegerField(min_value=1000,initial=10000)
    train_size = forms.FloatField(min_value=0.0,initial=0.5)

class NameModelMultipleChoiceField(forms.ModelMultipleChoiceField):
    def label_from_instance(self, obj):
        return obj.name

class NameModelChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        return obj.name

class TestingForm(forms.Form):
    algorithm = NameModelChoiceField(required=True, queryset=Algorithm.objects.all(),
                widget=forms.Select(attrs={'id':'algo-select'}), empty_label=None)

    dataset = NameModelChoiceField(required=True,queryset=Dataset.objects.all().order_by('name'),
                widget=forms.Select(attrs={'id':'datasets-select'}), empty_label=None)

    capital = forms.IntegerField(min_value=1000,initial=10000)

    train_size = forms.FloatField(min_value=0.0, initial = 0.5)
    load_parameters = forms.BooleanField(required=False, label='Load')
    save_parameters = forms.BooleanField(required=False, label='Save')

    def clean(self):
        cleaned_data = super(TestingForm, self).clean()
        algorithm = cleaned_data.get("algorithm")
        train_size = cleaned_data.get("train_size")

        if algorithm == Algorithm.objects.get(filename='lines_strat.py'):
            if train_size < 0.1 or train_size >= 1:
                raise forms.ValidationError('Algorithm Lines_strat should have train_size in range [0.1,1)')
        if algorithm == Algorithm.objects.get(filename='net4.py'):
            if train_size != 3000:
                raise forms.ValidationError('Algorithm Vanilla RNN should be tested with train size = 3000')
            dataset = cleaned_data.get('dataset')
            if dataset.name != 'CUR_EUR':
                raise forms.ValidationError('Algorithm Vanilla RNN should be tested on CUR_EUR dataset')
            load_flag = cleaned_data.get('load_parameters')
            if not load_flag:
                raise forms.ValidationError('For the algorithm Vanilla RNN train parameters should be loaded')
        if algorithm == Algorithm.objects.get(filename = 'brand_new.py'):
            if train_size != 0:
                raise forms.ValidationError('Algorithm Brand New should be tested with train size = 0')
            dataset = cleaned_data.get('dataset')
            if dataset.name != 'Brand_New_Data':
                raise forms.ValidationError('Algorithm Brand New should be tested on Brand_New_Data dataset')
            load_flag = cleaned_data.get('load_parameters')
            if not load_flag:
                raise forms.ValidationError('For the algorithm Brand New train parameters should be loaded')





