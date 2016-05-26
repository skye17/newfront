from django.conf.urls import url
from django.conf import settings
from django.views.static import serve

from . import views

urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'test/$', views.TestFormView.as_view(),name='testing'),
    url(r'about/$', views.AboutView.as_view(), name='about'),
    url(r'team/$', views.TeamView.as_view(), name='team'),
    url(r'add/$', views.AddView.as_view(), name = 'add'),
    url(r'archive/$', views.ArchiveView.as_view(), name = 'archive'),
    url(r'quick_start/$', views.QuickStartView.as_view(), name='quick_start'),
    url(r'^datasetinfo/(?P<pk>\d+)$', views.DatasetInfoView.as_view(), name = 'dataset_info'),
    url(r'datasets/$', views.DatasetListView.as_view(), name = 'datasets'),
    url(r'algorithms/$', views.AlgoListView.as_view(), name = 'algorithms'),
    url(r'add_dataset/$', views.UploadDatasetFormView.as_view(), name = 'add_dataset'),
    url(r'add_algorithm/$', views.UploadAlgorithmFormView.as_view(), name = 'add_algorithm'),
    url(r'delete_algorithm/(?P<pk>\d+)$', views.AlgorithmDeleteView.as_view(), name = 'delete_algorithm'),
    url(r'update_algorithm/(?P<pk>\d+)$', views.AlgorithmUpdateView.as_view(), name = 'update_algorithm'),
    url(r'test_results/(?P<pk>\d+)/$', views.TestResult.as_view(), name='test_results'),
    url(r'delete_algo/(?P<algo_pk>\d+)/$', views.delete_algo, name='delete_algo'),
    url(r'delete_dataset/(?P<data_pk>\d+)/$', views.delete_dataset, name='delete_dataset'),
    url(r'download_algo/(?P<algo_pk>\d+)/$', views.download_algo, name='download_algo'),
    url(r'download_params/(?P<algo_pk>\d+)/$', views.download_params, name='download_params'),
    url(r'loading/$', views.LoadingView.as_view(),name='loading'),

]

if settings.DEBUG:
    urlpatterns += [url(r'uploads/(?P<path>.*)',serve, {'document_root': settings.MEDIA_ROOT})]
