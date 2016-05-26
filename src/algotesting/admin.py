from django.contrib import admin
from models import Dataset, Algorithm, ExperimentResult

# Register your models here.
@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
	pass

@admin.register(Algorithm)
class AlgorithmAdmin(admin.ModelAdmin):
	pass

@admin.register(ExperimentResult)
class ExperimentResultAdmin(admin.ModelAdmin):
	pass


