# Generated by Django 4.1.2 on 2023-06-08 14:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0008_remove_userformalinformal_bebefore_formal_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Warning',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('formal_warning_count', models.IntegerField()),
                ('voca_warning_count', models.IntegerField()),
            ],
        ),
    ]
