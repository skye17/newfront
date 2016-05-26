from django.db import models
import os,shutil

upload_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'uploads/')

class Algorithm(models.Model):
    filename = models.CharField(default='', max_length=20, unique=True)
    name = models.CharField(default='Some algorithm', max_length=20)
    description = models.TextField(blank=True)
    algorithm = models.FileField(default='',upload_to='algorithms/')
    train_parameters = models.FileField(blank=True, null=True, upload_to='algorithms/train_params/')
    @models.permalink
    def get_absolute_url(self):
        return ('algotesting:algorithms',)

class Dataset(models.Model):
    name = models.CharField(default='Dataset', max_length=50, unique=True)
    data = models.FileField(default= '', upload_to='datasets/')
    data_filenames = models.CharField(default='',max_length=300)
    description = models.TextField(blank=True)
    frequency = models.CharField(default='daily', max_length= 6, choices=[('daily', 'daily'), ('minute', 'minute')])
    columns = models.CharField(default='', max_length=100)
    date_format = models.CharField(blank=True, default="", max_length=50)
    def delete(self, *args,**kwargs):
        path = os.path.join(upload_path, 'datasets/'+self.name)
        shutil.rmtree(path,ignore_errors = True)
        super(Dataset,self).delete(*args, **kwargs)


class ExperimentResult(models.Model):
    info = models.TextField(default='')
    test_results = models.TextField(default='')
    profit = models.BooleanField()
    results_path = models.CharField(default='', max_length=100)
    def delete(self, *args,**kwargs):
        os.remove(os.path.join(upload_path, self.results_path))
        super(ExperimentResult,self).delete(*args, **kwargs)
    @models.permalink
    def get_absolute_url(self):
        return ('algotesting:test_results', [str(self.pk)])
