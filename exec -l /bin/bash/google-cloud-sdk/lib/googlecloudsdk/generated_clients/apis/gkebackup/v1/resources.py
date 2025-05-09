# -*- coding: utf-8 -*- #
# Copyright 2023 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resource definitions for Cloud Platform Apis generated from apitools."""

import enum


BASE_URL = 'https://gkebackup.googleapis.com/v1/'
DOCS_URL = 'https://cloud.google.com/kubernetes-engine/docs/add-on/backup-for-gke'


class Collections(enum.Enum):
  """Collections for all supported apis."""

  PROJECTS = (
      'projects',
      'projects/{projectsId}',
      {},
      ['projectsId'],
      True
  )
  PROJECTS_LOCATIONS = (
      'projects.locations',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_BACKUPCHANNELS = (
      'projects.locations.backupChannels',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/backupChannels/'
              '{backupChannelsId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_BACKUPCHANNELS_BACKUPPLANASSOCIATIONS = (
      'projects.locations.backupChannels.backupPlanAssociations',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/backupChannels/'
              '{backupChannelsId}/backupPlanAssociations/'
              '{backupPlanAssociationsId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_BACKUPCHANNELS_BACKUPPLANBINDINGS = (
      'projects.locations.backupChannels.backupPlanBindings',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/backupChannels/'
              '{backupChannelsId}/backupPlanBindings/{backupPlanBindingsId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_BACKUPPLANS = (
      'projects.locations.backupPlans',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/backupPlans/'
              '{backupPlansId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_BACKUPPLANS_BACKUPS = (
      'projects.locations.backupPlans.backups',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/backupPlans/'
              '{backupPlansId}/backups/{backupsId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_BACKUPPLANS_BACKUPS_VOLUMEBACKUPS = (
      'projects.locations.backupPlans.backups.volumeBackups',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/backupPlans/'
              '{backupPlansId}/backups/{backupsId}/volumeBackups/'
              '{volumeBackupsId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_OPERATIONS = (
      'projects.locations.operations',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/operations/'
              '{operationsId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_RESTORECHANNELS = (
      'projects.locations.restoreChannels',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/restoreChannels/'
              '{restoreChannelsId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_RESTORECHANNELS_RESTOREPLANASSOCIATIONS = (
      'projects.locations.restoreChannels.restorePlanAssociations',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/restoreChannels/'
              '{restoreChannelsId}/restorePlanAssociations/'
              '{restorePlanAssociationsId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_RESTORECHANNELS_RESTOREPLANBINDINGS = (
      'projects.locations.restoreChannels.restorePlanBindings',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/restoreChannels/'
              '{restoreChannelsId}/restorePlanBindings/'
              '{restorePlanBindingsId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_RESTOREPLANS = (
      'projects.locations.restorePlans',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/restorePlans/'
              '{restorePlansId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_RESTOREPLANS_RESTORES = (
      'projects.locations.restorePlans.restores',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/restorePlans/'
              '{restorePlansId}/restores/{restoresId}',
      },
      ['name'],
      True
  )
  PROJECTS_LOCATIONS_RESTOREPLANS_RESTORES_VOLUMERESTORES = (
      'projects.locations.restorePlans.restores.volumeRestores',
      '{+name}',
      {
          '':
              'projects/{projectsId}/locations/{locationsId}/restorePlans/'
              '{restorePlansId}/restores/{restoresId}/volumeRestores/'
              '{volumeRestoresId}',
      },
      ['name'],
      True
  )

  def __init__(self, collection_name, path, flat_paths, params,
               enable_uri_parsing):
    self.collection_name = collection_name
    self.path = path
    self.flat_paths = flat_paths
    self.params = params
    self.enable_uri_parsing = enable_uri_parsing
