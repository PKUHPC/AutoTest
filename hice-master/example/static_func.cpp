#include <iostream>
#include <string>

#include <memory>

using namespace std;

#define LOG(str) \
  std::cout << "line " << __LINE__ << ": " << str << std::endl;
class Foo {

public:
  Foo() {
    LOG("Foo");
  }

  ~Foo() {
    LOG("~Foo");
  }

};

int main() {
  shared_ptr<Foo> foo = make_shared<Foo>(new Foo());

  LOG((bool)foo);
  foo.reset();
  LOG((bool)foo);
  foo = make_shared<Foo>(Foo());
  LOG((bool)foo);
  foo = nullptr;
  LOG((bool)foo);

  return 0;
}