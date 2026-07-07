{{- define "name" -}}
{{- default .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "fullname" -}}
{{- $name := default .Chart.Name -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Render a list of environment variables from a map of key-value pairs.

Possible values for the map are:
- string: the value is rendered as a string
    example:
      key: value
- map: the value is rendered as a YAML object
    example:
      key:
        valueFrom:
          secretKeyRef:
            name: secret-name
            key: secret-key
*/}}
{{- define "toEnv" }}
{{- range $key, $value := . }}
    {{- if or (kindIs "string" $value) (kindIs "int" $value) }}
- name: "{{ $key }}"
  value: {{ tpl $value $ | quote }}
    {{- else if kindIs "map" $value}}
- name: "{{ $key }}"
{{ $value | toYaml | indent 2}}
    {{- end }}
    {{- end }}
{{- end }}