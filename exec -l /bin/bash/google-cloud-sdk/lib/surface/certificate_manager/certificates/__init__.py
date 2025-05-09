# -*- coding: utf-8 -*- #
# Copyright 2020 Google Inc. All Rights Reserved.
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
"""The certificates command group for the Certificate Manager."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.calliope import base

_UNIVERSE_ADDITIONAL_INFO_MESSAGE = """\
          Only self-managed certificates are supported. Managed certificates are not
          currently supported.
          """


@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA,
                    base.ReleaseTrack.GA)
class CertificateMaps(base.Group):
  """Manage Certificate Manager certificates.

  Commands for managing certificates.
  """
  detailed_help = {
      'UNIVERSE ADDITIONAL INFO': _UNIVERSE_ADDITIONAL_INFO_MESSAGE,
  }
