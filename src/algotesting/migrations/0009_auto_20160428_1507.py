# -*- coding: utf-8 -*-
# Generated by Django 1.9.5 on 2016-04-28 15:07
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('algotesting', '0008_remove_dataset_data_path'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='columns',
            field=models.CharField(default=b'', max_length=200),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='name',
            field=models.CharField(default=b'', max_length=20, unique=True),
        ),
    ]
