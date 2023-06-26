# Generated by Django 4.1.2 on 2023-06-07 12:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0003_user'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserFormalInformal',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('room', models.CharField(max_length=1000000)),
                ('user', models.CharField(max_length=1000000)),
                ('formal_count', models.IntegerField()),
                ('formal_percent_avg', models.IntegerField()),
                ('informal_count', models.IntegerField()),
                ('informal_percent_avg', models.IntegerField()),
            ],
        ),
    ]