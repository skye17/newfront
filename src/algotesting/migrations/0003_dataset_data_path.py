# -*- coding: utf-8 -*-
# Generated by Django 1.9.5 on 2016-04-21 14:41
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('algotesting', '0002_dataset_data'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='data_path',
            field=models.CharField(default=b'', max_length=50),
        ),
    ]
