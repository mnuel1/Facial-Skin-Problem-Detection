import 'package:flutter/material.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: SplashScreen(),
    );
  }
}

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;

  @override
  void initState() {
    super.initState();

    _animationController = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );

    _animationController.forward(); // Start the animation

    Future.delayed(Duration(seconds: 2), () {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => HomeScreen()),
      );
    });
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: FadeTransition(
          opacity: Tween(begin: 0.0, end: 1.0).animate(
            CurvedAnimation(
              curve: Curves.easeIn,
              parent: _animationController,
            ),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              // Logo
              Image.asset('assets/image 9.png', height: 100, width: 100),
              SizedBox(height: 16.0), // Spacer
              // Text
              Text(
                'Descry',
                style: TextStyle(
                  fontSize: 36.0,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey, // Set text color to grey
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;

  final List<Widget> _pages = [
    BlankPage(title: 'Home'),
    BlankPage(title: 'Scan'),
    BlankPage(title: 'Remedies'),
    BlankPage(title: 'Profile'),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Center(child: Image.asset('assets/image 9.png', height: 30, width: 30)), // Logo
      ),
      body: _pages[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: '',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.scanner),
            label: '',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.healing),
            label: '',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person),
            label: '',
          ),
        ],
        currentIndex: _currentIndex,
        selectedItemColor: Colors.blue,
        unselectedItemColor: Colors.black,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
      ),
    );
  }
}

class BlankPage extends StatelessWidget {
  final String title;

  const BlankPage({required this.title});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
      ),
      body: Center(
        child: Text('$title Page'),
      ),
    );
  }
}
