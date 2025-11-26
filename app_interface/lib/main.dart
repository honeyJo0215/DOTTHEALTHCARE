import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const FoodCameraApp());
}

class FoodCameraApp extends StatelessWidget {
  const FoodCameraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DOTTHEALTH Food Camera (beta)',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.orange),
      ),
      home: const FoodCameraPage(),
    );
  }
}

class FoodCameraPage extends StatefulWidget {
  const FoodCameraPage({super.key});

  @override
  State<FoodCameraPage> createState() => _FoodCameraPageState();
}

class _FoodCameraPageState extends State<FoodCameraPage> {
  final TextEditingController _serverController = TextEditingController(
    // 나중에 실제 IP로 바꿔줘 (예: http://192.168.0.5:8000)
    text: 'http://127.0.0.1:8000',
  );

  File? _imageFile;
  bool _isLoading = false;

  List<dynamic> _predictions = [];
  Map<String, dynamic>? _subPrediction;
  List<dynamic> _candidates = [];

  double? _selectedCalories;

  final ImagePicker _picker = ImagePicker();

  Future<void> _pickFromCamera() async {
    final XFile? picked =
        await _picker.pickImage(source: ImageSource.camera, imageQuality: 85);

    if (picked != null) {
      setState(() {
        _imageFile = File(picked.path);
        _predictions = [];
        _subPrediction = null;
        _candidates = [];
        _selectedCalories = null;
      });
    }
  }

  Future<void> _pickFromGallery() async {
    final XFile? picked =
        await _picker.pickImage(source: ImageSource.gallery, imageQuality: 85);

    if (picked != null) {
      setState(() {
        _imageFile = File(picked.path);
        _predictions = [];
        _subPrediction = null;
        _candidates = [];
        _selectedCalories = null;
      });
    }
  }

  Future<void> _sendToServer() async {
    if (_imageFile == null) return;

    final baseUrl = _serverController.text.trim();
    if (!baseUrl.startsWith('http')) {
      _showError('서버 주소를 올바르게 입력해주세요.\n예: http://192.168.0.5:8000');
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final uri = Uri.parse('$baseUrl/predict?topk=3&food_topk=5');
      final request = http.MultipartRequest('POST', uri);

      request.files.add(
        await http.MultipartFile.fromPath('image', _imageFile!.path),
      );

      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        setState(() {
          _predictions = (data['predictions'] ?? []) as List<dynamic>;
          _subPrediction =
              data['sub_prediction'] != null ? data['sub_prediction'] as Map<String, dynamic> : null;
          _candidates = (data['candidates'] ?? []) as List<dynamic>;

          // 기본 선택 칼로리: 첫 번째 candidate의 calories
          if (_candidates.isNotEmpty) {
            _selectedCalories = (_candidates.first['calories'] as num?)?.toDouble();
          }
        });
      } else {
        _showError('서버 오류: ${response.statusCode}\n${response.body}');
      }
    } catch (e) {
      _showError('요청 실패: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg)),
    );
  }

  void _onSelectCandidate(Map<String, dynamic> cand) {
    setState(() {
      _selectedCalories = (cand['calories'] as num?)?.toDouble();
    });
  }

  void _onConfirmIntake() {
    if (_selectedCalories == null) {
      _showError('칼로리를 선택하거나 입력해주세요.');
      return;
    }
    // TODO: 여기서 실제 "오늘 섭취 칼로리" 관리 앱으로 반환하는 로직을 붙이면 됨.
    // 예: Navigator.pop(context, _selectedCalories);
    _showError('오늘 섭취 칼로리에 ${_selectedCalories!.round()} kcal 반영했다고 가정!');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Food Camera (beta)'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(32),
          child: Padding(
            padding: const EdgeInsets.only(bottom: 8.0),
            child: Text(
              'DOTTHEALTH · 로컬 서버용',
              style: Theme.of(context).textTheme.labelSmall,
            ),
          ),
        ),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(12),
          children: [
            // 서버 주소 입력
            TextField(
              controller: _serverController,
              decoration: const InputDecoration(
                labelText: '서버 주소 (PC IP:포트)',
                hintText: '예: http://192.168.0.5:8000',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),

            // 이미지 프리뷰 + 버튼
            Row(
              children: [
                Expanded(
                  child: _imageFile == null
                      ? Container(
                          height: 160,
                          alignment: Alignment.center,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: Colors.grey),
                          ),
                          child: const Text('사진을 선택하거나 촬영하세요'),
                        )
                      : ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.file(
                            _imageFile!,
                            height: 160,
                            fit: BoxFit.cover,
                          ),
                        ),
                ),
                const SizedBox(width: 8),
                Column(
                  children: [
                    ElevatedButton.icon(
                      onPressed: _pickFromCamera,
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('촬영'),
                    ),
                    const SizedBox(height: 8),
                    OutlinedButton.icon(
                      onPressed: _pickFromGallery,
                      icon: const Icon(Icons.photo_library),
                      label: const Text('갤러리'),
                    ),
                    const SizedBox(height: 8),
                    FilledButton(
                      onPressed: _isLoading ? null : _sendToServer,
                      child: _isLoading
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Text('칼로리 분석'),
                    ),
                  ],
                ),
              ],
            ),

            const SizedBox(height: 16),

            // 예측 결과
            if (_predictions.isNotEmpty || _subPrediction != null)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('이미지 분류 결과',
                      style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 8),
                  if (_subPrediction != null)
                    Text(
                      '세부라벨: ${_subPrediction!['sub_label_display']} '
                      '(${(_subPrediction!['confidence'] as num).toStringAsFixed(2)})',
                    ),
                  for (final p in _predictions)
                    Text(
                      '- ${p['label_display']} '
                      '(${(p['confidence'] as num).toStringAsFixed(2)})',
                    ),
                  const Divider(height: 24),
                ],
              ),

            // food_db 후보 + 칼로리 선택
            if (_candidates.isNotEmpty)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('food_db 후보',
                      style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 8),
                  for (final c in _candidates)
                    ListTile(
                      title: Text(c['name']),
                      subtitle: Text(
                        '${c['calories']} kcal / '
                        '${c['serving_size']} ${c['unit']} '
                        '(score: ${(c['match_score'] as num).toStringAsFixed(2)})',
                      ),
                      trailing: Radio<double>(
                        value: (c['calories'] as num).toDouble(),
                        groupValue: _selectedCalories,
                        onChanged: (v) {
                          if (v != null) {
                            _onSelectCandidate(c);
                          }
                        },
                      ),
                      onTap: () => _onSelectCandidate(c),
                    ),
                  const SizedBox(height: 12),
                  // 직접 입력 칼로리
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          '선택 칼로리: '
                          '${_selectedCalories != null ? _selectedCalories!.round().toString() : "-"} kcal',
                        ),
                      ),
                      FilledButton(
                        onPressed: _onConfirmIntake,
                        child: const Text('오늘 섭취 칼로리에 반영'),
                      ),
                    ],
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
