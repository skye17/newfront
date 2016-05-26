from django.shortcuts import redirect, get_object_or_404
from django.views.generic import TemplateView, DetailView, ListView, FormView, UpdateView, DeleteView
from django.core.urlresolvers import reverse
from django.http.response import HttpResponseRedirect,HttpResponse
from forms import QuickStartForm, UploadDataFilesForm, TestingForm, UploadAlgorithmFileForm
from models import Dataset, ExperimentResult, Algorithm
from code.sample import handle_algorithm_testing

import os, mimetypes,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FILES_DELIMITER = ';'
DEFAULT_DATASET = 'Default_Daily'
DEFAULT_ALGORITHM = 'user_defined.py'

class IndexView(TemplateView):
    template_name = "index.html"

class LoadingView(TemplateView):
    template_name = "loading1.html"

class AboutView(TemplateView):
    template_name = "about.html"

class TeamView(TemplateView):
    template_name = "team.html"

class AddView(TemplateView):
    template_name = "add.html"

class ArchiveView(TemplateView):
    template_name = "archive.html"

class DatasetInfoView(DetailView):
    model = Dataset
    template_name = 'algotesting/dataset_page.html'

class DatasetListView(ListView):
    template_name = 'algotesting/datasets.html'
    form_class = UploadDataFilesForm
    context_object_name = 'datasets'
    def get_queryset(self):
        return Dataset.objects.all().order_by('name')

    def get_context_data(self, **kwargs):
        context = super(DatasetListView, self).get_context_data(**kwargs)
        context['form'] = UploadDataFilesForm
        return context

class AlgoListView(ListView):
    template_name = 'algotesting/algorithms.html'
    form_class = UploadAlgorithmFileForm
    context_object_name = 'algorithms'
    paginate_by = 20
    def get_queryset(self):
        return Algorithm.objects.all()

    def get_context_data(self, **kwargs):
        context = super(AlgoListView, self).get_context_data(**kwargs)
        context['form'] = UploadAlgorithmFileForm
        return context

def download_algo(request,algo_pk):
    algo = get_object_or_404(Algorithm, id=algo_pk)
    algo_filename = algo.filename.replace('algorithms/', '')
    fileurl=algo.algorithm.url
    mimetype = mimetypes.guess_type(fileurl)
    response = HttpResponse(algo.algorithm, content_type=mimetype)
    response['Content-Disposition'] = 'attachment; filename={}'.format(algo_filename)
    return response

def download_params(request,algo_pk):
    algo = get_object_or_404(Algorithm, id=algo_pk)
    params_filename = algo.train_parameters.name.replace('algorithms/train_params/', '')
    fileurl=algo.train_parameters.url
    mimetype = mimetypes.guess_type(fileurl)
    print mimetype
    response = HttpResponse(algo.train_parameters, content_type=mimetype)
    response['Content-Disposition'] = 'attachment; filename={}'.format(params_filename)
    return response

def delete_algo(request, algo_pk):
    algo = get_object_or_404(Algorithm, id=algo_pk)
    if algo:
        Algorithm.objects.get(pk=algo_pk).delete()
    return HttpResponseRedirect(reverse("algotesting:algorithms"))

def delete_dataset(request, data_pk):
    dataset = get_object_or_404(Dataset, id=data_pk)
    if dataset:
        Dataset.objects.get(pk=data_pk).delete()
    return HttpResponseRedirect(reverse("algotesting:datasets"))

class AlgorithmDeleteView(DeleteView):
    model = Algorithm
    template_name = "algotesting/delete_algorithm.html"
    def get_success_url(self):
        return reverse('algotesting:algorithms')

class AlgorithmUpdateView(UpdateView):
    model = Algorithm
    form_class = UploadAlgorithmFileForm
    template_name = "algotesting/update_algorithm.html"
    def form_valid(self, form):
        d = form.save(commit=False)
        d.save()
        d.filename= d.algorithm.name.replace('algorithms/','')
        d.save()
        return super(AlgorithmUpdateView, self).form_valid(form)

class UploadDatasetFormView(FormView):
    form_class = UploadDataFilesForm
    template_name = "algotesting/upload_dataset.html"
    def form_valid(self, form):
        d = form.save(commit=True)
        filepath = os.path.join(BASE_DIR,'uploads/'+d.data.name)
        dir_path = os.path.join(BASE_DIR,'uploads/datasets/'+d.name)
        os.mkdir(dir_path)
        if filepath.endswith('.zip'):
            ## Dealing with zip archive
            archive = zipfile.ZipFile(filepath)
            d.data_filenames = FILES_DELIMITER.join(archive.namelist())
            archive.extractall(path=dir_path)
            os.remove(filepath)
        else:
            file = form.cleaned_data['data']
            for file_line in file:
                d.columns = file_line[:-1]
                break
            print os.path.join(dir_path,file.name)
            os.rename(filepath,os.path.join(dir_path,file.name))
            filename = d.data.name.replace('datasets/','')
            d.data_filenames = filename
        d.save()
        return super(UploadDatasetFormView, self).form_valid(form)
    def get_success_url(self):
        return reverse('algotesting:datasets')


class UploadAlgorithmFormView(FormView):
    form_class = UploadAlgorithmFileForm
    template_name = "algotesting/upload_algorithm.html"

    def get_context_data(self, **kwargs):
        context = super(UploadAlgorithmFormView, self).get_context_data(**kwargs)
        context['object'] = Algorithm.objects.get(filename=DEFAULT_ALGORITHM)
        return context

    def form_valid(self, form):
        d = form.save(commit=False)
        d.save()
        d.filename= d.algorithm.name.replace('algorithms/','')
        d.save()
        return super(UploadAlgorithmFormView, self).form_valid(form)

    def get_success_url(self):
        return reverse('algotesting:algorithms')


class QuickStartView(FormView):
    form_class = QuickStartForm
    template_name = "algotesting/quick_start.html"
    def form_valid(self, form):
        algorithm = Algorithm.objects.get(filename=DEFAULT_ALGORITHM)
        data = Dataset.objects.get(name=DEFAULT_DATASET)
        names = [data.name+'/'+ file_name for file_name in data.data_filenames.split(FILES_DELIMITER)]
        capital = form.cleaned_data['capital']
        train_size = form.cleaned_data['train_size']
        params = {'algorithm':algorithm.filename, 'data_filenames': names, 'frequency':data.frequency,
                      'base_capital':capital, 'parser':data.date_format, 'train_size':train_size,
                      'load_file':'', 'save_file':''}
        result,portfolio = handle_algorithm_testing(params)
        earned = portfolio.iloc[-1] - portfolio.iloc[0]
        portfolio_info = 'Earned money = {}\nMax portfolio value = {}\n' \
                         'Min portfolio value = {}\n'.format(earned,portfolio.max(),
                portfolio.min())
        info = ''

        for key, value in params.items():
            if value:
                if type(value) is list:
                    if key != 'data_filenames':
                        info += "{}:{}\n".format(key, ', '.join(map(str, value)))
                    else:
                        info += "{}:{}\n".format('datasets', ', '.join(map(lambda x: x.name,[data])))
                else:
                    info += "{}:{}\n".format(key,value)

        experiment_info = info
        profit = earned > 0
        experiment_result = ExperimentResult(info = experiment_info,
                                             test_results = portfolio_info,
                                             profit = profit,results_path = '')
        experiment_result.save()

        relative_path = 'testing/results/quick_start.pdf'
        graph_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                      'uploads/'+relative_path)

        result.savefig(graph_path)
        plt.close(result)

        experiment_result.results_path = relative_path
        experiment_result.save()

        return redirect(experiment_result)

class TestFormView(FormView):
    form_class = TestingForm
    template_name = "algotesting/test.html"

    def get_context_data(self, **kwargs):
        context = super(TestFormView, self).get_context_data(**kwargs)
        context['pk'] = Dataset.objects.get(name=DEFAULT_DATASET).pk
        return context

    def form_valid(self, form):
        algorithm = form.cleaned_data['algorithm']
        print 'Algorithm = {}, {}'.format(algorithm.pk, algorithm.name)
        dataset = form.cleaned_data['dataset']
        load = form.cleaned_data['load_parameters']
        save = form.cleaned_data['save_parameters']
        load_params = ''
        save_params = ''
        if load and algorithm.train_parameters:
            load_params = algorithm.train_parameters.name
        if save:
            save_params = algorithm.train_parameters.name

        names = [dataset.name+'/'+ file_name for file_name in dataset.data_filenames.split(FILES_DELIMITER)]
        frequency = dataset.frequency
        parser = dataset.date_format

        print 'Data file names = {}'.format(names)

        capital = form.cleaned_data['capital']
        train_size = form.cleaned_data['train_size']
        params = {'algorithm':algorithm.filename,'data_filenames': names, 'frequency':frequency,
                      'base_capital':capital, 'parser':parser, 'train_size':train_size,
                      'load_file':load_params, 'save_file':save_params}
        result, portfolio = handle_algorithm_testing(params)
        earned = portfolio.iloc[-1] - portfolio.iloc[0]
        portfolio_info = 'Earned money = {}\nMax portfolio value = {}\n' \
                         'Min portfolio value = {}\n'.format(earned,portfolio.max(),
                portfolio.min())
        info = ''
        for key, value in params.items():
            if value:
                if type(value) is list:
                    if key != 'data_filenames':
                        info += "{}:{}\n".format(key, ', '.join(map(str, value)))
                    else:
                        info += "{}:{}\n".format('datasets', ', '.join(map(lambda x: x.name,[dataset])))
                else:
                    info += "{}:{}\n".format(key,value)

        experiment_info = info
        profit = earned > 0
        experiment_result = ExperimentResult(info = experiment_info,
                                             test_results = portfolio_info,
                                             profit = profit,results_path = '')
        experiment_result.save()
        relative_path = 'testing/results/graph'+str(experiment_result.pk)+'.pdf'
        graph_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                      'uploads/'+relative_path)
        result.savefig(graph_path)
        plt.close(result)
        experiment_result.results_path = relative_path
        experiment_result.save()
        return redirect(experiment_result)


class TestResult(DetailView):
    model = ExperimentResult
    template_name = 'algotesting/test_results.html'